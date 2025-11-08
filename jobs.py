#!/usr/bin/python3
# -*- coding: utf-8 -*-


import asyncio
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

from config import logger


class JobStatusEnum(Enum):
    CANCELED = "canceled"
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    DONE = "done"


class AsyncJobQueue:
    """
    Queue that accepts jobs (each job can contain multiple locations).
    A job is processed by a worker thread which executes analyses for each
    location (parallelized with a small ThreadPoolExecutor per-job).
    """

    def __init__(
        self,
        worker_callable: Callable[..., Dict[str, Any]],
        max_workers: int = 2,
        per_job_concurrency: int = 20,
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

        self._jobs: Dict[str, Dict] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = True
        self._workers: List[asyncio.Task] = []

    async def start(self):
        """Start the job queue workers"""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(), name=f"job-worker-{i}")
            self._workers.append(worker)
        logger.info(f"Started {self.max_workers} job workers")

    async def stop(self):
        """Stop all job workers"""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, payload: dict) -> str:
        """
        Submit a job payload. payload should include:
          - 'locations': list of location dicts (each with lat, lng, storefront_direction, day, time)
          - other optional: proxy, user info...
        Returns: job_id
        """
        job_id = os.urandom(8).hex()
        job_record = {
            "status": JobStatusEnum.PENDING,
            "payload": payload,
            "result": {"count": 0, "locations": []},
            "error": None,
            "completed": 0,
            "failure": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "cancel_requested": False,
            "_logged_to_db": False,
        }
        self._jobs[job_id] = job_record
        await self._queue.put(job_id)
        return job_id

    async def get(self, job_id: str) -> Optional[dict]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    async def remove(self, job_id: str) -> None:
        """Remove job from tracking"""
        self._jobs.pop(job_id, None)

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

    async def _worker_loop(self):
        """Worker loop to process jobs"""
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_job(job_id)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                continue

    async def _process_job(self, job_id: str):
        """Process a single job"""
        job = self._jobs.get(job_id)
        if not job:
            return

        job["status"] = JobStatusEnum.RUNNING
        job["updated_at"] = time.time()

        locations = job["payload"].get("locations", [])[:20]
        results: List[Any] = [None] * len(locations)

        try:
            # Process locations concurrently with limited concurrency
            semaphore = asyncio.Semaphore(self.per_job_concurrency)

            async def process_location(idx: int, loc: dict):
                if job.get("cancel_requested"):
                    return None

                async with semaphore:
                    try:
                        return await self._run_single_location(
                            loc, job["payload"].get("proxy")
                        )
                    except Exception as e:
                        job["failure"] += 1
                        error_msg = f"Location analysis failed: {str(e)}"
                        logger.error(f"Failed location {idx}: {error_msg}")
                        return {"error": error_msg}

            # Create tasks for all locations
            tasks = []
            for idx, loc in enumerate(locations):
                if job.get("cancel_requested"):
                    break
                task = asyncio.create_task(process_location(idx, loc))
                tasks.append((idx, task))

            # Collect results as they complete
            for idx, task in tasks:
                if job.get("cancel_requested"):
                    break

                try:
                    res = await task
                    if res:
                        # Handle screenshot URL generation
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
                                logger.warning(
                                    f"Failed to generate screenshot URL: {e}"
                                )

                        res["screenshot_url"] = screenshot_url
                        results[idx] = res

                except Exception as e:
                    logger.error(f"Task processing failed: {e}")
                finally:
                    job["completed"] += 1

            if not job.get("cancel_requested"):
                job["result"]["locations"] = [
                    r for r in results if r and "error" not in r
                ]
                job["result"]["count"] = len(job["result"]["locations"])

                # Check if all locations failed
                if len(locations) > 0 and job.get("failure", 0) == len(locations):
                    error_messages = [
                        r.get("error", "Unknown error")
                        for r in results
                        if r and "error" in r
                    ]
                    job["error"] = (
                        f"All {len(locations)} location(s) failed. Errors: {'; '.join(error_messages[:3])}"
                    )
                    job["status"] = JobStatusEnum.FAILED
                else:
                    job["status"] = JobStatusEnum.DONE

        except Exception as e:
            error_msg = f"Job processing failed: {str(e)}"
            job["error"] = error_msg
            job["status"] = JobStatusEnum.FAILED
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            job["updated_at"] = time.time()

    async def _run_single_location(self, loc: dict, proxy: Optional[str] = None):
        """Execute worker for a single location"""
        return await self.worker_callable(
            loc.get("lat"),
            loc.get("lng"),
            loc.get("storefront_direction", "north"),
            loc.get("day"),
            loc.get("time"),
            proxy,
        )


# For backward compatibility
JobQueue = AsyncJobQueue
