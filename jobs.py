#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import queue
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Callable, Dict, List
from urllib.parse import urljoin

from config import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class JobStatusEnum(Enum):
    CANCELED = "canceled"
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    DONE = "done"


class JobID:
    @staticmethod
    def choice(seq):
        """Return a secure random element from a non-empty sequence."""
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        # generate random index securely
        idx = int.from_bytes(os.urandom(4), "big") % len(seq)
        return seq[idx]

    @staticmethod
    def random_string(length=12, alphabet=string.ascii_letters + string.digits):
        """Generate a secure random string from given alphabet."""
        return "".join(JobID.choice(alphabet) for _ in range(length))


class JobQueue:
    """
    Queue that accepts jobs (each job can contain multiple locations).
    A job is processed by a worker thread which executes analyses for each
    location (parallelized with a small ThreadPoolExecutor per-job).
    """

    def __init__(
        self,
        worker_callable: Callable[..., Dict[str, Any]],
        max_workers: int = 20,
        per_job_concurrency: int = 4,
    ):
        """
        Args:
            worker_callable: callable that performs a *single* location analysis (synchronous).
                             signature: worker_callable(lat, lng, storefront_direction, day, time, proxy) -> dict
            max_workers: number of background job worker threads (how many jobs processed concurrently).
            per_job_concurrency: how many locations inside a single job are processed concurrently.
        """
        self.worker_callable = worker_callable
        self.max_workers = max_workers
        self.per_job_concurrency = per_job_concurrency

        self._queue = queue.Queue()
        self._jobs: Dict[str, Dict] = {}
        self._start_workers()

    def _start_workers(self):
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()

    def submit(self, payload: dict) -> str:
        """
        Submit a job payload. payload should include:
          - 'locations': list of location dicts (each with lat, lng, storefront_direction, day, time)
          - other optional: proxy, user info...
        Returns: job_id
        """
        # job_id = str(uuid.uuid4())
        job_id = JobID.random_string()
        job_record = {
            "status": JobStatusEnum.PENDING,
            "payload": payload,
            "result": {"count": 0, "results": []},
            "error": None,
            "remaining": 0,
            "failure": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "cancel_requested": False,
        }
        self._jobs[job_id] = job_record
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> dict:
        return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        try:
            self._jobs.pop(job_id)
        except Exception:
            pass

    def cancel(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None

        if job["status"] in (
            JobStatusEnum.DONE,
            JobStatusEnum.FAILED,
            JobStatusEnum.CANCELED,
        ):
            return job

        job["cancel_requested"] = True
        job["status"] = JobStatusEnum.CANCELED
        job["updated_at"] = time.time()
        return job

    def _worker_loop(self):
        while True:
            job_id = self._queue.get()
            job = self._jobs.get(job_id)

            if not job:
                self._queue.task_done()
                continue

            job["status"] = JobStatusEnum.RUNNING
            job["updated_at"] = time.time()

            locations = job["payload"].get("locations", [])[:20]  # enforce limit
            results: List[Any] = [None] * len(locations)

            job["remaining"] = len(locations)

            try:
                # process locations in parallel but limited concurrency per job
                with ThreadPoolExecutor(
                    max_workers=max(1, self.per_job_concurrency)
                ) as ex:
                    future_to_index = {}
                    for idx, loc in enumerate(locations):
                        if job.get("cancel_requested"):
                            break

                        future = ex.submit(
                            self._run_single_location, loc, job["payload"].get("proxy")
                        )
                        future_to_index[future] = idx

                    # collect results as they complete, keep order via index
                    for future in as_completed(future_to_index):
                        if job.get("cancel_requested"):
                            break

                        idx = future_to_index[future]
                        try:
                            res = future.result()
                            if res and "error" in res:
                                raise ValueError("There is no analysis data")
                        except Exception as e:
                            job["failure"] += 1

                            error_msg = f"Location analysis failed: {str(e)}"
                            res = {"error": error_msg}
                            logger.error(
                                f"Failed to analyze location at index {idx}: {error_msg}"
                            )

                        screenshot_path = res.get("screenshot_path") or res.get(
                            "pinned_screenshot_path"
                        )
                        screenshot_url = None

                        if screenshot_path:
                            try:
                                rel = os.path.relpath(
                                    screenshot_path, os.path.abspath("static")
                                )
                                screenshot_url = urljoin(
                                    job["payload"].get("request_base_url", "/"),
                                    f"static/{rel.replace(os.sep, '/')}",
                                )
                            except Exception as e:
                                error_msg = f"Fetching Screenshot failed: {str(e)}"
                                logger.warning(
                                    f"Failed to fetch location screenshot {idx}: {error_msg}"
                                )

                        res["screenshot_url"] = screenshot_url
                        results[idx] = res

                        job["remaining"] -= 1

                if not job.get("cancel_requested"):
                    job["result"].update({"count": len(locations), "results": results})
                    job["status"] = JobStatusEnum.DONE

                    # Check if all locations failed
                    if len(locations) > 0 and job.get("failure", 0) == len(locations):
                        # All locations failed - mark job as FAILED
                        error_messages = [
                            r.get("error", "Unknown error")
                            for r in results
                            if r and "error" in r
                        ]
                        job["error"] = (
                            f"All {len(locations)} location(s) failed. Errors: {'; '.join(error_messages[:3])}"  # Limit to first 3 errors
                        )
                        job["status"] = JobStatusEnum.FAILED
                    else:
                        job["status"] = JobStatusEnum.DONE

            except Exception as e:
                error_msg = f"The job failed: {str(e)}"
                job["error"] = error_msg
                job["status"] = JobStatusEnum.FAILED
                logger.warning(f"Failed to complete The Job {idx}: {str(e)}")

            finally:
                job["updated_at"] = time.time()
                self._queue.task_done()

    # Optional retry wrapper for the worker callable (exposed small retries for robustness)
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(Exception),
    )
    def _call_worker(self, *args, **kwargs):
        return self.worker_callable(*args, **kwargs)

    def _run_single_location(self, loc: dict, proxy: str | None):
        """
        Execute the provided worker callable for a single location dict.
        location dict expected keys:
          lat, lng, storefront_direction, day, time
        """

        # call the worker callable (this may raise)
        return self._call_worker(
            loc.get("lat"),
            loc.get("lng"),
            loc.get("storefront_direction", "north"),
            loc.get("day"),
            loc.get("time"),
            proxy,
        )
