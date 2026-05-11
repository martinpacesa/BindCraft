#!/usr/bin/env python3
"""
FastAPI server for BindCraft peptide design inference.
Provides REST API endpoints for:
- Peptide design from target structure
- Structure prediction
- Binding affinity scoring
- Batch processing
"""

import os
import sys
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths
WORKSPACE = Path("/workspace")
DATA_DIR = Path("/data")
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_DIR = Path(tempfile.gettempdir()) / "bindcraft_api"

# Create directories
for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="BindCraft API",
    description="Peptide design via AlphaFold2 + MPNN + PyRosetta",
    version="1.0.0",
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class DesignRequest(BaseModel):
    """Request model for binder design."""
    target_pdb_file: str = Field(..., description="Target PDB filename in /data/inputs/")
    binder_name: str = Field(default="binder", description="Prefix for output files")
    target_chains: str = Field(default="A", description="Target chains (e.g., 'A' or 'A,B')")
    target_hotspot: Optional[str] = Field(default=None, description="Target residues (e.g., '1-10' or 'A1-10')")
    binder_lengths: str = Field(default="35-50", description="Binder length range (e.g., '35-50')")
    num_designs: int = Field(default=100, description="Number of design trajectories")
    filters: str = Field(default="default", description="Filter preset: default, strict, or relaxed")


class ValidationRequest(BaseModel):
    """Request model for structure validation."""
    pdb_files: List[str] = Field(..., description="List of PDB filenames in /data/inputs/")


class BatchDesignRequest(BaseModel):
    """Request model for batch processing multiple targets."""
    targets: List[DesignRequest]


class DesignResponse(BaseModel):
    """Response model for design results."""
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    target: str
    designs: List[Dict[str, Any]] = Field(default=[], description="List of designed binders")
    metrics: Dict[str, Any] = Field(default={})
    output_dir: str


# ============================================================================
# Dummy Pipeline (replace with actual BindCraft calls)
# ============================================================================

class PeptideDesignPipeline:
    """Wrapper around BindCraft functionality."""
    
    def __init__(self, workspace: Path = WORKSPACE):
        self.workspace = workspace
        self.bindcraft_dir = workspace / "bindcraft"
        self.mpnn_dir = workspace / "ProteinMPNN"
        logger.info(f"Pipeline initialized at {self.bindcraft_dir}")
    
    def validate_pdb(self, pdb_path: Path) -> bool:
        """Validate PDB file format."""
        if not pdb_path.exists():
            return False
        
        try:
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("pdb", str(pdb_path))
            return len(structure) > 0
        except Exception as e:
            logger.error(f"PDB validation failed: {e}")
            return False
    
    async def design_peptide(self, design_req: DesignRequest) -> DesignResponse:
        """
        Design peptide binders against target structure.
        This is a placeholder - actual implementation calls BindCraft bindcraft.py
        """
        job_id = f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = OUTPUT_DIR / job_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Verify input PDB exists
        target_pdb = INPUT_DIR / design_req.target_pdb_file
        if not target_pdb.exists():
            raise HTTPException(status_code=400, detail=f"Target PDB not found: {design_req.target_pdb_file}")
        
        if not self.validate_pdb(target_pdb):
            raise HTTPException(status_code=400, detail="Invalid PDB file format")
        
        logger.info(f"[{job_id}] Starting design for {target_pdb.name}")
        
        # TODO: Call actual BindCraft pipeline
        # For now, return dummy response
        response = DesignResponse(
            job_id=job_id,
            status="queued",
            target=design_req.target_pdb_file,
            designs=[],
            metrics={
                "target": design_req.target_pdb_file,
                "chains": design_req.target_chains,
                "binder_length_range": design_req.binder_lengths,
                "num_trajectories": design_req.num_designs,
                "status": "queued",
            },
            output_dir=str(output_path),
        )
        
        return response
    
    async def validate_structures(self, pdb_files: List[str]) -> Dict[str, Any]:
        """Validate multiple PDB files."""
        results = {}
        
        for filename in pdb_files:
            filepath = INPUT_DIR / filename
            if filepath.exists():
                is_valid = self.validate_pdb(filepath)
                results[filename] = {
                    "valid": is_valid,
                    "path": str(filepath),
                }
            else:
                results[filename] = {
                    "valid": False,
                    "error": "File not found",
                }
        
        return results


# Initialize pipeline
pipeline = PeptideDesignPipeline()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "workspace": str(WORKSPACE),
        "gpu_available": check_cuda(),
    }


def check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


@app.get("/info")
async def get_info():
    """Get system and model information."""
    try:
        import torch
        cuda_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        if torch.cuda.is_available():
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        cuda_info = {"available": False, "error": "PyTorch not installed or CUDA unavailable"}
    
    return {
        "app_name": "BindCraft API",
        "version": "1.0.0",
        "workspace": str(WORKSPACE),
        "data_dir": str(DATA_DIR),
        "cuda": cuda_info,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/design")
async def design_peptide(request: DesignRequest, background_tasks: BackgroundTasks):
    """
    Design peptide binders against a target structure.
    
    - **target_pdb_file**: PDB filename (must be in /data/inputs/)
    - **binder_name**: Prefix for output files
    - **target_chains**: Which chains to target (e.g., "A" or "A,B")
    - **target_hotspot**: Optional: specific residues to target
    - **binder_lengths**: Length range (e.g., "35-50")
    - **num_designs**: Number of design trajectories (default 100)
    
    Returns job_id for async polling.
    """
    try:
        response = await pipeline.design_peptide(request)
        logger.info(f"Design job created: {response.job_id}")
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Design failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate_structures(request: ValidationRequest):
    """
    Validate multiple PDB structure files.
    
    - **pdb_files**: List of filenames in /data/inputs/
    
    Returns validation results for each file.
    """
    try:
        results = await pipeline.validate_structures(request.pdb_files)
        return {"validations": results}
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_jobs():
    """List all design jobs and their status."""
    jobs = []
    for job_dir in OUTPUT_DIR.glob("design_*"):
        job_info = {
            "job_id": job_dir.name,
            "created": job_dir.stat().st_mtime,
            "path": str(job_dir),
        }
        
        # Check for result file
        result_file = job_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                job_info["results"] = json.load(f)
        
        jobs.append(job_info)
    
    return {"jobs": jobs}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific design job."""
    job_dir = OUTPUT_DIR / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    result_file = job_dir / "results.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    
    return {
        "job_id": job_id,
        "status": "running",
        "output_dir": str(job_dir),
    }


@app.get("/jobs/{job_id}/download")
async def download_results(job_id: str):
    """Download results as ZIP file."""
    job_dir = OUTPUT_DIR / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Create zip file
    zip_path = TEMP_DIR / f"{job_id}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', job_dir)
    
    return FileResponse(
        path=zip_path,
        filename=f"{job_id}.zip",
        media_type="application/zip",
    )


@app.post("/upload")
async def upload_pdb(file: UploadFile = File(...)):
    """Upload a PDB file for processing."""
    try:
        # Save file
        filepath = INPUT_DIR / file.filename
        content = await file.read()
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        # Validate
        is_valid = pipeline.validate_pdb(filepath)
        
        return {
            "filename": file.filename,
            "path": str(filepath),
            "size": len(content),
            "valid": is_valid,
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebUI static files (optional)
# ============================================================================

@app.get("/")
async def root():
    """Serve API documentation."""
    return {
        "message": "BindCraft API Server",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    # Run with: python api_server.py
    # Or in Docker: uvicorn api_server:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
