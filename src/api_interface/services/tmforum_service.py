"""
TMForum-compliant AI Inference Job Management Service (TMF922).

This service implements the business logic for TMForum standard AI inference operations,
providing a standardized interface for AI model inference jobs.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ..models.tmforum_models import (
    TMForumInferenceJob,
    TMForumInferenceJobCreate,
    TMForumInferenceJobUpdate,
    TMForumInferenceJobState,
    TMForumInferenceJobPriority,
    TMForumInferenceInput,
    TMForumInferenceOutput,
    TMForumInferenceModel
)
from ..services.prediction_service import prediction_service
from ..utils.logging import logger


class TMForumInferenceJobManager:
    """Manages TMForum AI Inference Jobs in memory."""

    def __init__(self):
        self.jobs: Dict[str, TMForumInferenceJob] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_jobs = 10  # Configurable limit

        # Start the job processor
        self.processor_task = None

    async def start_processor(self):
        """Start the background job processor."""
        if self.processor_task is None:
            self.processor_task = asyncio.create_task(self._process_jobs())
            logger.info("TMForum job processor started")

    async def stop_processor(self):
        """Stop the background job processor."""
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
            logger.info("TMForum job processor stopped")

    async def _process_jobs(self):
        """Background job processor."""
        while True:
            try:
                job_id = await self.job_queue.get()
                job = self.jobs.get(job_id)

                if job and job.state == TMForumInferenceJobState.ACKNOWLEDGED:
                    # Start processing this job
                    task = asyncio.create_task(self._execute_job(job_id))
                    self.processing_tasks[job_id] = task

                    # Limit concurrent jobs
                    if len(self.processing_tasks) >= self.max_concurrent_jobs:
                        # Wait for a task to complete
                        done, pending = await asyncio.wait(
                            self.processing_tasks.values(),
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        # Clean up completed tasks
                        for task in done:
                            for jid, t in list(self.processing_tasks.items()):
                                if t == task:
                                    del self.processing_tasks[jid]
                                    break

                self.job_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job processor: {str(e)}")

    async def _execute_job(self, job_id: str):
        """Execute a single inference job."""
        try:
            job = self.jobs.get(job_id)
            if not job:
                return

            # Update job state to in progress
            job.state = TMForumInferenceJobState.IN_PROGRESS
            job.startDate = datetime.utcnow()
            logger.info(f"Started processing TMForum job: {job_id}")

            # Extract text from input
            text = job.input.inputData.get("text", "")
            if not text:
                raise ValueError("No text provided in input data")

            # Perform prediction using existing service
            from ..models.request_models import PredictionRequest, ModelType
            from ..models.response_models import PredictionResponse

            request = PredictionRequest(
                text=text,
                model=ModelType.OTS_MBERT
            )

            response: PredictionResponse = await prediction_service.predict(request)

            # Create output
            output_data = {
                "label": response.label.value,
                "probability": response.probability
            }

            output = TMForumInferenceOutput(
                outputType="classification",
                outputFormat="json",
                outputData=output_data,
                confidence=response.probability,
                outputMetadata={
                    "model_used": response.model_info.name,
                    "model_version": response.model_info.version,
                    "processing_time_seconds": response.processing_time
                }
            )

            # Update job with results
            job.state = TMForumInferenceJobState.COMPLETED
            job.output = output
            job.completionDate = datetime.utcnow()
            job.processingTimeMs = int(response.processing_time * 1000)

            logger.info(f"Completed TMForum job: {job_id}, result: {output_data['label']}")

        except Exception as e:
            # Update job with error
            job = self.jobs.get(job_id)
            if job:
                job.state = TMForumInferenceJobState.FAILED
                job.errorMessage = str(e)
                job.completionDate = datetime.utcnow()
                logger.error(f"Failed TMForum job: {job_id}, error: {str(e)}")

        finally:
            # Clean up processing task
            if job_id in self.processing_tasks:
                del self.processing_tasks[job_id]


class TMForumService:
    """TMForum-compliant AI Inference Job Management Service."""

    def __init__(self):
        self.job_manager = TMForumInferenceJobManager()

    async def initialize(self):
        """Initialize the service."""
        await self.job_manager.start_processor()

    async def shutdown(self):
        """Shutdown the service."""
        await self.job_manager.stop_processor()

    async def create_inference_job(self, request: TMForumInferenceJobCreate) -> TMForumInferenceJob:
        """
        Create a new inference job.

        Args:
            request: Job creation request

        Returns:
            Created job with ID and initial state
        """
        job_id = str(uuid.uuid4())

        # Get the actual model version from the loaded model
        from ..services.prediction_service import get_model_version
        actual_model_version = get_model_version()

        # Update model info with correct version
        updated_model = request.model.model_copy()
        updated_model.version = actual_model_version

        # Create the job
        job = TMForumInferenceJob(
            id=job_id,
            href=f"/tmf-api/aiInferenceJob/{job_id}",
            state=TMForumInferenceJobState.ACKNOWLEDGED,
            priority=request.priority,
            input=request.input,
            model=updated_model,
            creationDate=datetime.utcnow(),
            name=request.name,
            description=request.description,
            category=request.category,
            externalId=request.externalId
        )

        # Store the job
        self.job_manager.jobs[job_id] = job

        # Queue for processing
        await self.job_manager.job_queue.put(job_id)

        logger.info(f"Created TMForum inference job: {job_id}")
        return job

    async def get_inference_job(self, job_id: str) -> Optional[TMForumInferenceJob]:
        """
        Get an inference job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job details or None if not found
        """
        return self.job_manager.jobs.get(job_id)

    async def list_inference_jobs(
        self,
        state: Optional[TMForumInferenceJobState] = None,
        priority: Optional[TMForumInferenceJobPriority] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[TMForumInferenceJob]:
        """
        List inference jobs with optional filtering.

        Args:
            state: Filter by job state
            priority: Filter by job priority
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of matching jobs
        """
        jobs = list(self.job_manager.jobs.values())

        # Apply filters
        if state:
            jobs = [job for job in jobs if job.state == state]
        if priority:
            jobs = [job for job in jobs if job.priority == priority]

        # Sort by creation date (newest first)
        jobs.sort(key=lambda x: x.creationDate or datetime.min, reverse=True)

        # Apply pagination
        start = offset
        end = offset + limit
        return jobs[start:end]

    async def update_inference_job(self, job_id: str, update: TMForumInferenceJobUpdate) -> Optional[TMForumInferenceJob]:
        """
        Update an inference job.

        Args:
            job_id: Job identifier
            update: Update request

        Returns:
            Updated job or None if not found
        """
        job = self.job_manager.jobs.get(job_id)
        if not job:
            return None

        # Validate update based on current state
        if job.state in [TMForumInferenceJobState.IN_PROGRESS, TMForumInferenceJobState.COMPLETED]:
            # Limited updates allowed for running/completed jobs
            if update.state and update.state != job.state:
                raise ValueError(f"Cannot change state of {job.state.value} job")
        else:
            # Allow more updates for pending jobs
            if update.state:
                job.state = update.state
            if update.priority:
                job.priority = update.priority

        # Always allow metadata updates
        if update.name is not None:
            job.name = update.name
        if update.description is not None:
            job.description = update.description

        logger.info(f"Updated TMForum inference job: {job_id}")
        return job

    async def delete_inference_job(self, job_id: str) -> bool:
        """
        Delete an inference job.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found or cannot be deleted
        """
        job = self.job_manager.jobs.get(job_id)
        if not job:
            return False

        # Only allow deletion of completed, failed, or cancelled jobs
        if job.state not in [
            TMForumInferenceJobState.COMPLETED,
            TMForumInferenceJobState.FAILED,
            TMForumInferenceJobState.CANCELLED
        ]:
            return False

        # Cancel processing if still running
        if job_id in self.job_manager.processing_tasks:
            task = self.job_manager.processing_tasks[job_id]
            task.cancel()
            del self.job_manager.processing_tasks[job_id]

        # Remove the job
        del self.job_manager.jobs[job_id]

        logger.info(f"Deleted TMForum inference job: {job_id}")
        return True

    def get_job_stats(self) -> Dict[str, Any]:
        """Get job processing statistics."""
        jobs = list(self.job_manager.jobs.values())
        states = defaultdict(int)

        for job in jobs:
            states[job.state.value] += 1

        return {
            "total_jobs": len(jobs),
            "jobs_by_state": dict(states),
            "processing_queue_size": self.job_manager.job_queue.qsize(),
            "active_processing_tasks": len(self.job_manager.processing_tasks)
        }


# Global TMForum service instance
tmforum_service = TMForumService()