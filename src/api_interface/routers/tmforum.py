"""
TMForum-compliant AI Inference Job Management API (TMF922) router.

This implements TMForum standard endpoints for AI inference services,
specifically designed for text classification tasks.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..models.tmforum_models import (
    TMForumInferenceJob,
    TMForumInferenceJobCreate,
    TMForumInferenceJobUpdate,
    TMForumInferenceJobList,
    TMForumInferenceJobState,
    TMForumInferenceJobPriority,
    TMForumInferenceInput,
    TMForumInferenceOutput,
    TMForumInferenceModel,
    TMForumError
)
from ..services.tmforum_service import tmforum_service
from ..middleware.security import verify_ip_address
from ..utils.logging import logger

router = APIRouter(
    prefix="/tmf-api/aiInferenceJob",
    tags=["TMForum AI Inference Job Management (TMF922)"],
    dependencies=[Depends(verify_ip_address)]
)


@router.post(
    "",
    response_model=TMForumInferenceJob,
    status_code=201,
    responses={
        201: {"model": TMForumInferenceJob, "description": "Inference job created successfully"},
        400: {"model": TMForumError, "description": "Bad Request"},
        403: {"model": TMForumError, "description": "Forbidden"},
        422: {"model": TMForumError, "description": "Validation Error"},
        500: {"model": TMForumError, "description": "Internal Server Error"},
    },
    summary="Create AI Inference Job",
    description="""
    Create a new AI inference job for text classification.

    This endpoint follows TMForum TMF922 standard for AI Inference Job Management.
    The job will be queued for processing and can be monitored using the returned job ID.
    """
)
async def create_inference_job(request: TMForumInferenceJobCreate) -> TMForumInferenceJob:
    """
    Create a new AI inference job.

    Args:
        request: Inference job creation request

    Returns:
        Created inference job with ID and initial state

    Raises:
        HTTPException: For validation or processing errors
    """
    try:
        logger.info(f"Creating TMForum inference job: {request.name or 'unnamed'}")

        # Validate input data
        if request.input.inputType != "text":
            raise HTTPException(
                status_code=400,
                detail=TMForumError(
                    code="INVALID_INPUT_TYPE",
                    reason="Invalid input type",
                    message="Only 'text' input type is supported for text classification",
                    status="400"
                ).model_dump()
            )

        if "text" not in request.input.inputData:
            raise HTTPException(
                status_code=400,
                detail=TMForumError(
                    code="MISSING_TEXT_DATA",
                    reason="Missing text data",
                    message="Input data must contain 'text' field",
                    status="400"
                ).model_dump()
            )

        # Create the job
        job = await tmforum_service.create_inference_job(request)

        logger.info(f"Created TMForum inference job: {job.id}")
        return job

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating TMForum inference job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumError(
                code="INTERNAL_ERROR",
                reason="Internal server error",
                message="Failed to create inference job",
                status="500"
            ).model_dump()
        )


@router.get(
    "/{job_id}",
    response_model=TMForumInferenceJob,
    responses={
        200: {"model": TMForumInferenceJob, "description": "Inference job retrieved successfully"},
        403: {"model": TMForumError, "description": "Forbidden"},
        404: {"model": TMForumError, "description": "Job Not Found"},
        500: {"model": TMForumError, "description": "Internal Server Error"},
    },
    summary="Retrieve AI Inference Job",
    description="""
    Retrieve an AI inference job by ID.

    Returns the current state and results of the inference job.
    If the job is completed, the output field will contain the classification results.
    """
)
async def get_inference_job(job_id: str) -> TMForumInferenceJob:
    """
    Retrieve an inference job by ID.

    Args:
        job_id: Unique identifier of the inference job

    Returns:
        Inference job details

    Raises:
        HTTPException: If job not found or access denied
    """
    try:
        logger.info(f"Retrieving TMForum inference job: {job_id}")

        job = await tmforum_service.get_inference_job(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail=TMForumError(
                    code="JOB_NOT_FOUND",
                    reason="Job not found",
                    message=f"Inference job {job_id} not found",
                    status="404"
                ).model_dump()
            )

        return job

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving TMForum inference job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumError(
                code="INTERNAL_ERROR",
                reason="Internal server error",
                message="Failed to retrieve inference job",
                status="500"
            ).model_dump()
        )


@router.get(
    "",
    response_model=TMForumInferenceJobList,
    responses={
        200: {"model": TMForumInferenceJobList, "description": "Inference jobs retrieved successfully"},
        403: {"model": TMForumError, "description": "Forbidden"},
        500: {"model": TMForumError, "description": "Internal Server Error"},
    },
    summary="List AI Inference Jobs",
    description="""
    List AI inference jobs with optional filtering.

    Supports filtering by state, priority, and date ranges.
    """
)
async def list_inference_jobs(
    state: Optional[TMForumInferenceJobState] = Query(None, description="Filter by job state"),
    priority: Optional[TMForumInferenceJobPriority] = Query(None, description="Filter by job priority"),
    limit: int = Query(50, description="Maximum number of jobs to return", ge=1, le=1000),
    offset: int = Query(0, description="Number of jobs to skip", ge=0)
) -> TMForumInferenceJobList:
    """
    List inference jobs with optional filtering.

    Args:
        state: Filter by job state
        priority: Filter by job priority
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip

    Returns:
        List of inference jobs
    """
    try:
        logger.info(f"Listing TMForum inference jobs: state={state}, priority={priority}, limit={limit}, offset={offset}")

        jobs = await tmforum_service.list_inference_jobs(
            state=state,
            priority=priority,
            limit=limit,
            offset=offset
        )

        return TMForumInferenceJobList(jobs=jobs)

    except Exception as e:
        logger.error(f"Error listing TMForum inference jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumError(
                code="INTERNAL_ERROR",
                reason="Internal server error",
                message="Failed to list inference jobs",
                status="500"
            ).model_dump()
        )


@router.patch(
    "/{job_id}",
    response_model=TMForumInferenceJob,
    responses={
        200: {"model": TMForumInferenceJob, "description": "Inference job updated successfully"},
        400: {"model": TMForumError, "description": "Bad Request"},
        403: {"model": TMForumError, "description": "Forbidden"},
        404: {"model": TMForumError, "description": "Job Not Found"},
        422: {"model": TMForumError, "description": "Validation Error"},
        500: {"model": TMForumError, "description": "Internal Server Error"},
    },
    summary="Update AI Inference Job",
    description="""
    Update an AI inference job.

    Allows updating job priority, state, and metadata.
    Note: Some fields may not be updatable once the job is in progress.
    """
)
async def update_inference_job(job_id: str, update: TMForumInferenceJobUpdate) -> TMForumInferenceJob:
    """
    Update an inference job.

    Args:
        job_id: Unique identifier of the inference job
        update: Fields to update

    Returns:
        Updated inference job

    Raises:
        HTTPException: If job not found or update not allowed
    """
    try:
        logger.info(f"Updating TMForum inference job: {job_id}")

        job = await tmforum_service.update_inference_job(job_id, update)
        if not job:
            raise HTTPException(
                status_code=404,
                detail=TMForumError(
                    code="JOB_NOT_FOUND",
                    reason="Job not found",
                    message=f"Inference job {job_id} not found",
                    status="404"
                ).model_dump()
            )

        return job

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating TMForum inference job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumError(
                code="INTERNAL_ERROR",
                reason="Internal server error",
                message="Failed to update inference job",
                status="500"
            ).model_dump()
        )


@router.delete(
    "/{job_id}",
    status_code=204,
    responses={
        204: {"description": "Inference job deleted successfully"},
        403: {"model": TMForumError, "description": "Forbidden"},
        404: {"model": TMForumError, "description": "Job Not Found"},
        409: {"model": TMForumError, "description": "Conflict - Job in progress"},
        500: {"model": TMForumError, "description": "Internal Server Error"},
    },
    summary="Delete AI Inference Job",
    description="""
    Delete an AI inference job.

    Only completed, failed, or cancelled jobs can be deleted.
    Running jobs must be cancelled first.
    """
)
async def delete_inference_job(job_id: str):
    """
    Delete an inference job.

    Args:
        job_id: Unique identifier of the inference job

    Raises:
        HTTPException: If job not found or cannot be deleted
    """
    try:
        logger.info(f"Deleting TMForum inference job: {job_id}")

        success = await tmforum_service.delete_inference_job(job_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=TMForumError(
                    code="JOB_NOT_FOUND",
                    reason="Job not found",
                    message=f"Inference job {job_id} not found",
                    status="404"
                ).model_dump()
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting TMForum inference job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumError(
                code="INTERNAL_ERROR",
                reason="Internal server error",
                message="Failed to delete inference job",
                status="500"
            ).model_dump()
        )