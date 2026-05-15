"""
Single GPU Worker launcher for KernelGym.
"""
import asyncio
import argparse
import logging
import sys
import redis.asyncio as redis

from kernelgym.config import settings
KEY_PREFIX = settings.redis_key_prefix
from kernelgym.config import setup_logging
from kernelgym.worker.gpu_worker import GPUWorker
from kernelgym.worker.gpu_background_compile_worker import BackgroundCompileGPUWorker

logger = logging.getLogger("kernelgym.single_worker")


async def main():
    """Main entry point for single GPU worker."""
    parser = argparse.ArgumentParser(description="Start a single GPU worker")
    parser.add_argument("--worker-id", required=True, help="Worker ID")
    parser.add_argument("--device", required=True, help="GPU device (e.g., cuda:0)")
    parser.add_argument("--persistent", action="store_true", help="Record process info for persistent monitor")
    args = parser.parse_args()
    
    # Configure logging
    logger = setup_logging(f"worker_{args.worker_id}")
    
    # Initialize Redis connection
    redis_client = redis.from_url(settings.redis_url)
    await redis_client.ping()
    logger.info(f"Redis connection established for worker {args.worker_id}")
    
    worker_class = str(getattr(settings, "gpu_worker_class", "legacy")).strip().lower()
    if worker_class == "legacy":
        legacy_split_mode = str(
            getattr(settings, "worker_split_compile_mode", "legacy")
        ).strip().lower()
        if legacy_split_mode == "background_compile":
            worker_class = "background_compile"
    if worker_class in {"legacy", "gpuworker"}:
        worker = GPUWorker(args.worker_id, args.device, redis_client)
    elif worker_class in {"background_compile", "background", "backgroundcompile"}:
        worker = BackgroundCompileGPUWorker(args.worker_id, args.device, redis_client)
    else:
        raise ValueError(
            f"Invalid GPU_WORKER_CLASS: {worker_class}. Expected legacy or background_compile."
        )
    logger.info(
        "Selected worker class %s for worker %s",
        worker.__class__.__name__,
        args.worker_id,
    )
    
    try:
        logger.info(f"Starting single worker {args.worker_id} on device {args.device}")
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
    finally:
        try:
            # In persistent mode, clear process info on clean exit
            if args.persistent:
                await redis_client.delete(f"{KEY_PREFIX}:worker_process:{args.worker_id}")
        except Exception:
            pass
        await worker.stop()
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
