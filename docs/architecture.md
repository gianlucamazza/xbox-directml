# System Architecture

This document outlines the architecture of the DirectML machine learning system for Xbox Series S.

## System Overview

The architecture integrates Microsoft components to create a functional machine learning pipeline on Xbox Series S hardware.

```
┌─────────────────────────────────────────────────────────┐
│                   Xbox Series S                          │
│                                                         │
│  ┌───────────────┐      ┌───────────────────────────┐  │
│  │ Applicazione  │      │   Risorse Hardware        │  │
│  │     UWP       │─────▶│                           │  │
│  └───────────────┘      │  ┌─────────────────────┐  │  │
│          │              │  │  GPU AMD RDNA 2     │  │  │
│          ▼              │  └─────────────────────┘  │  │
│  ┌───────────────┐      │  ┌─────────────────────┐  │  │
│  │   DirectX 12  │─────▶│  │  8GB RAM condivisa  │  │  │
│  └───────────────┘      │  └─────────────────────┘  │  │
│          │              │  ┌─────────────────────┐  │  │
│          ▼              │  │  CPU Zen 2 8-core   │  │  │
│  ┌───────────────┐      │  └─────────────────────┘  │  │
│  │   DirectML    │─────▶│                           │  │
│  └───────────────┘      └───────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### UWP Application
The Universal Windows Platform (UWP) application serves as the container for our machine learning solution. It provides:
- Cross-compatibility between Xbox and Windows
- User interface for model selection and execution
- Integration with Xbox ecosystem

### DirectX 12
DirectX 12 provides low-level access to the graphics hardware, enabling:
- Efficient memory management
- Command queue control
- Resource creation and binding

### DirectML
DirectML is Microsoft's dedicated API for machine learning acceleration that:
- Provides hardware-accelerated operators
- Integrates with DirectX 12
- Optimizes execution for the specific hardware

### Xbox Series S Hardware
The hardware components of the Xbox Series S that are leveraged:
- AMD RDNA 2 GPU: Handles parallel computation for ML inference
- 8-core Zen 2 CPU: Manages preprocessing and application logic
- 8GB Shared RAM: Stores application, model weights, and intermediate tensors

## Communication Flow

1. The UWP application initializes DirectML through DirectX 12
2. Model weights are loaded from storage into GPU-accessible memory
3. Input data is preprocessed on CPU then transferred to GPU memory
4. DirectML executes the computational graph on the GPU
5. Results are transferred back to CPU-accessible memory
6. The application processes and displays the results

## Memory Management

Memory management is critical due to the limited 8GB shared RAM:

| Component | Memory Allocation Strategy |
|-----------|----------------------------|
| System OS | ~2GB reserved              |
| UWP App   | ~1GB for application code  |
| Models    | Dynamic allocation based on model size |
| Tensors   | Pooled allocation to minimize fragmentation |
| Results   | Minimal footprint with immediate processing |

## Concurrency Model

The system uses the following concurrency approach:
- UI runs on the main thread
- Data preprocessing occurs on background CPU threads
- DirectML operations execute asynchronously on the GPU
- Results processing happens on a dedicated thread
- Resource loading uses async patterns to prevent UI blocking 