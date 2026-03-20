"""Cheap rate-limit probe for autointerp LLM providers.

Designed to find a practical throughput ceiling without paying for full
autointerp runs. It sends small structured-output requests and reports
success rate, latency, achieved throughput, and retryable failures.

Example:
    python -m spd.autointerp.scripts.probe_rate_limits \\
        --config_json '{"type":"google_ai","model":"gemini-3-flash-preview"}' \\
        --rpm_values 300,600,1200 \\
        --concurrency_values 16,32,64,128 \\
        --n_requests 120
"""

import asyncio
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from pydantic import TypeAdapter

from spd.autointerp.providers import LLMConfig, RetryableAPIError, create_provider

_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "reasoning"],
}


@dataclass
class ProbeStats:
    rpm: int
    concurrency: int
    n_requests: int
    n_success: int
    n_retryable: int
    n_json_errors: int
    n_other_errors: int
    wall_time_s: float
    input_tokens: int
    output_tokens: int
    latencies_s: list[float]

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_requests if self.n_requests else 0.0

    @property
    def achieved_rpm(self) -> float:
        return self.n_requests * 60 / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def achieved_qps(self) -> float:
        return self.n_requests / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def p50_latency_s(self) -> float:
        return statistics.median(self.latencies_s) if self.latencies_s else 0.0

    @property
    def p95_latency_s(self) -> float:
        if not self.latencies_s:
            return 0.0
        if len(self.latencies_s) == 1:
            return self.latencies_s[0]
        return statistics.quantiles(self.latencies_s, n=20)[18]

    @property
    def tpm(self) -> float:
        return self.input_tokens * 60 / self.wall_time_s if self.wall_time_s > 0 else 0.0

    def to_row(self) -> str:
        return (
            f"{self.rpm:>5} rpm  "
            f"{self.concurrency:>4} conc  "
            f"success={self.n_success:>4}/{self.n_requests:<4}  "
            f"retryable={self.n_retryable:<3}  "
            f"json={self.n_json_errors:<3}  "
            f"other={self.n_other_errors:<3}  "
            f"succ_rate={self.success_rate:>6.1%}  "
            f"qps={self.achieved_qps:>5.2f}  "
            f"achieved_rpm={self.achieved_rpm:>6.1f}  "
            f"p50={self.p50_latency_s:>4.2f}s  "
            f"p95={self.p95_latency_s:>5.2f}s  "
            f"tpm={self.tpm:>8.0f}"
        )


def _parse_int_list(raw: str | tuple[Any, ...] | list[Any]) -> list[int]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    values = [int(part) for part in parts]
    assert values, f"Expected at least one integer in {raw!r}"
    return values


def _build_prompt(i: int) -> str:
    return (
        "Return strict JSON with exactly two fields: "
        '"label" and "reasoning". '
        'Set label to "ok". '
        f'Set reasoning to "probe-{i}".'
    )


async def _run_probe_point(
    config: LLMConfig,
    rpm: int,
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    timeout_ms: int,
) -> ProbeStats:
    provider = create_provider(config)
    limiter = AsyncLimiter(max_rate=rpm, time_period=60)
    queue: asyncio.Queue[int | None] = asyncio.Queue()

    n_success = 0
    n_retryable = 0
    n_json_errors = 0
    n_other_errors = 0
    input_tokens = 0
    output_tokens = 0
    latencies_s: list[float] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal n_success, n_retryable, n_json_errors, n_other_errors
        nonlocal input_tokens, output_tokens
        while (idx := await queue.get()) is not None:
            await limiter.acquire()
            start = time.perf_counter()
            try:
                response = await provider.chat(
                    prompt=_build_prompt(idx),
                    max_tokens=max_tokens,
                    response_schema=_SCHEMA,
                    timeout_ms=timeout_ms,
                )
                parsed = json.loads(response.content)
                assert parsed["label"] == "ok"
                assert parsed["reasoning"] == f"probe-{idx}"
                elapsed = time.perf_counter() - start
                async with lock:
                    n_success += 1
                    input_tokens += response.input_tokens
                    output_tokens += response.output_tokens
                    latencies_s.append(elapsed)
            except RetryableAPIError:
                elapsed = time.perf_counter() - start
                async with lock:
                    n_retryable += 1
                    latencies_s.append(elapsed)
            except json.JSONDecodeError:
                elapsed = time.perf_counter() - start
                async with lock:
                    n_json_errors += 1
                    latencies_s.append(elapsed)
            except Exception:
                elapsed = time.perf_counter() - start
                async with lock:
                    n_other_errors += 1
                    latencies_s.append(elapsed)

    start = time.perf_counter()
    try:
        for i in range(n_requests):
            await queue.put(i)
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
    finally:
        await provider.close()
    wall_time_s = time.perf_counter() - start

    return ProbeStats(
        rpm=rpm,
        concurrency=concurrency,
        n_requests=n_requests,
        n_success=n_success,
        n_retryable=n_retryable,
        n_json_errors=n_json_errors,
        n_other_errors=n_other_errors,
        wall_time_s=wall_time_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latencies_s=latencies_s,
    )


async def _run_matrix(
    config: LLMConfig,
    rpm_values: Iterable[int],
    concurrency_values: Iterable[int],
    n_requests: int,
    max_tokens: int,
    timeout_ms: int,
    stop_on_retryable: bool,
    success_rate_floor: float,
) -> list[ProbeStats]:
    results: list[ProbeStats] = []
    for concurrency in concurrency_values:
        for rpm in rpm_values:
            print(f"\n=== probing rpm={rpm}, concurrency={concurrency} ===")
            stats = await _run_probe_point(
                config=config,
                rpm=rpm,
                concurrency=concurrency,
                n_requests=n_requests,
                max_tokens=max_tokens,
                timeout_ms=timeout_ms,
            )
            print(stats.to_row())
            results.append(stats)
            if stop_on_retryable and (
                stats.n_retryable > 0 or stats.success_rate < success_rate_floor
            ):
                print(
                    "stopping sweep after limit hit "
                    f"(retryable={stats.n_retryable}, success_rate={stats.success_rate:.1%})"
                )
                return results
    return results


def main(
    config_json: dict[str, Any],
    rpm_values: str = "300,600,1200",
    concurrency_values: str = "16,32,64,128",
    n_requests: int = 120,
    max_tokens: int = 128,
    timeout_ms: int = 30000,
    stop_on_retryable: bool = True,
    success_rate_floor: float = 0.98,
) -> None:
    """Run a cheap provider rate-limit sweep."""
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    load_dotenv()

    config = TypeAdapter(LLMConfig).validate_python(config_json)
    rpm_list = _parse_int_list(rpm_values)
    concurrency_list = _parse_int_list(concurrency_values)

    print("provider_config=", config.model_dump())
    print("rpm_values=", rpm_list)
    print("concurrency_values=", concurrency_list)
    print("n_requests=", n_requests)
    print("max_tokens=", max_tokens)
    print("timeout_ms=", timeout_ms)

    results = asyncio.run(
        _run_matrix(
            config=config,
            rpm_values=rpm_list,
            concurrency_values=concurrency_list,
            n_requests=n_requests,
            max_tokens=max_tokens,
            timeout_ms=timeout_ms,
            stop_on_retryable=stop_on_retryable,
            success_rate_floor=success_rate_floor,
        )
    )

    print("\n=== summary ===")
    for stats in results:
        print(stats.to_row())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
