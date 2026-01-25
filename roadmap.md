# `mlrs` Library Development Roadmap

This document outlines the tasks to evolve `mlrs` from a specialized 2D matrix autograd engine into a scalable, general-purpose tensor library. Each item includes the goal, the reasoning, and an example of the proposed structure.

## Phase 1: Core Refactoring

### [ ] 1. Use Generics for Data Types

-   **Goal:** Allow `Tensor` to hold different numeric types (`f32`, `f64`, `i32`, etc.).
-   **Why:** This increases the library's versatility for a wider range of applications beyond `f32`.
-   **Implementation:**
    -   Modify `Tensor` and `Matrix` (or its `Tensor` replacement) to be generic over a new `Numeric` trait.
    -   The `Numeric` trait will bound types to those that can be used in tensor computations (e.g., `Copy`, `Clone`, `Debug`, `PartialEq`, `Add`, `Mul`, etc.).

**Example:**
```rust
// In a new types.rs or similar
pub trait Numeric:
    Copy
    + Clone
    + std::fmt::Debug
    + PartialEq
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    // ... other necessary traits
{
    // Associated constants or methods if needed
}

impl Numeric for f32 {}
impl Numeric for f64 {}
// ... impl for other types

// In tensor.rs
struct Tensor<T: Numeric> {
    data: Vec<T>,
    shape: Vec<usize>,
    // ... other fields
}
```

### [ ] 2. Generalize `Matrix` to an N-dimensional `Tensor`

-   **Goal:** Move from a 2D `Matrix` to a proper N-dimensional `Tensor` struct.
-   **Why:** This is the most critical step for scalability and a prerequisite for implementing modern ML operations like convolutions.
-   **Implementation:**
    -   Replace `rows` and `cols` with a `shape: Vec<usize>`.
    -   Implement a striding system for efficient memory access without data duplication for operations like slicing or transposing.

**Example:**
```rust
// In tensor.rs
struct Tensor<T: Numeric> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>, // For efficient indexing
    // ...
}

impl<T: Numeric> Tensor<T> {
    fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides = Self::calculate_strides(&shape);
        Self { data, shape, strides }
    }

    // ... other methods like `get`, `set`, `transpose`, `slice`
}
```

## Phase 2: Scalability and Extensibility

### [ ] 3. Introduce a `Backend` Trait for Computation

-   **Goal:** Decouple tensor logic from the computation implementation (CPU, GPU).
-   **Why:** This makes the library scalable to new hardware platforms and allows for different optimization strategies.
-   **Implementation:**
    -   Define a `Backend` trait that declares all computational kernels (e.g., `add`, `mul`, `matmul`).
    -   The `Tensor` will be associated with a backend.
    -   Create a `CPUBackend` that contains the current `rayon`-based logic.

**Example:**
```rust
// In backend.rs
pub trait Backend {
    type DType: Numeric;

    fn add(a: &Tensor<Self::DType>, b: &Tensor<Self::DType>) -> Result<Tensor<Self::DType>, Error>;
    fn matmul(a: &Tensor<Self::DType>, b: &Tensor<Self::DType>) -> Result<Tensor<Self::DType>, Error>;
    // ... other operations
}

// In cpu_backend.rs
pub struct CPUBackend;

impl Backend for CPUBackend {
    type DType = f32; // Or could be generic itself

    fn add(...) {
        // CPU implementation
    }
    fn matmul(...) {
        // rayon-based implementation
    }
}
```

### [ ] 4. Decouple Operations with a Trait-based System

-   **Goal:** Replace the monolithic `Op` enum and `match` statement with a modular system for defining operations.
-   **Why:** This will make the library much easier to extend with new operations without touching the core autograd engine.
-   **Implementation:**
    -   Define an `Operation` trait with `forward` and `backward` methods.
    -   Each operation (e.g., `Add`, `Mul`, `ReLU`) will be a struct that implements this trait.
    -   The autograd engine will store `Box<dyn Operation>` in the graph nodes.

**Example:**
```rust
// In ops/mod.rs
pub trait Operation<B: Backend> {
    fn forward(&self, inputs: &[&Tensor<B>]) -> Tensor<B>;
    fn backward(&self, output_grad: &Tensor<B>) -> Vec<Tensor<B>>;
}

// In ops/add.rs
pub struct Add;

impl<B: Backend> Operation<B> for Add {
    fn forward(...) { /* ... */ }
    fn backward(...) { /* ... */ }
}
```

## Phase 3: Integration and Usability

### [ ] 5. Implement Robust Serialization

-   **Goal:** Safely and efficiently save and load tensors and entire models.
-   **Why:** This is essential for model persistence, transfer learning, and production deployment.
-   **Implementation:**
    -   Use the `serde` framework to derive `Serialize` and `Deserialize` for `Tensor` and other relevant structs.
    -   Choose a format like `bincode` for performance or JSON/RON for human-readability.

**Example:**
```rust
// In tensor.rs
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Tensor<T: Numeric> {
    // ... fields
}

// Usage
fn save_tensor<T: Numeric + serde::Serialize>(tensor: &Tensor<T>, path: &str) {
    // ... use serde and a format to write to file
}
```

### [ ] 6. Create a Dedicated FFI Layer for Integration

-   **Goal:** Expose a stable C-compatible API for using `mlrs` from other languages like Python.
-   **Why:** This is the key to "easy integration" and allows `mlrs` to be the performant backend for a higher-level library.
-   **Implementation:**
    -   Create an `ffi` module with `extern "C"` functions.
    -   Use opaque pointers (`*mut Tensor`) to represent `mlrs` objects.
    -   Provide functions for creating, deleting, and operating on these objects.

**Example:**
```rust
// In ffi.rs
use crate::tensor::Tensor;

#[no_mangle]
pub extern "C" fn tensor_create(data: *const f32, shape: *const usize, ndim: usize) -> *mut Tensor<f32> {
    // ... create a Tensor and return a pointer
}

#[no_mangle]
pub extern "C" fn tensor_free(ptr: *mut Tensor<f32>) {
    // ... free the memory
}

#[no_mangle]
pub extern "C" fn tensor_add(a: *const Tensor<f32>, b: *const Tensor<f32>) -> *mut Tensor<f32> {
    // ...
}
```
