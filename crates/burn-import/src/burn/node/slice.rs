#![allow(clippy::needless_range_loop)]

use super::NodeCodegen;
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

#[derive(Debug, Clone)]
pub struct SliceNode {
    pub input: Type,
    pub output: Type,
    pub starts: SliceParam,
    pub ends: SliceParam,
    pub axes: Option<SliceParam>,
}

#[derive(Debug, Clone)]
pub enum SliceParam {
    Static(Vec<i64>),
    Runtime(Type),
}

impl SliceNode {
    pub fn new(input: Type, output: Type, starts: SliceParam, ends: SliceParam) -> Self {
        Self {
            input,
            output,
            starts,
            ends,
            axes: None,
        }
    }

    pub fn with_axes(mut self, axes: SliceParam) -> Self {
        self.axes = Some(axes);
        self
    }

    fn generate_slice(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();

        match &self.input {
            Type::Tensor(tensor) => {
                self.generate_tensor_slice(tensor, scope, node_position, output)
            }
            Type::Shape(shape) => self.generate_shape_slice(shape, output),
            _ => panic!("Unsupported input type for SliceNode"),
        }
    }

    fn generate_tensor_slice(
        &self,
        tensor: &crate::burn::TensorType,
        scope: &mut Scope,
        node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        let input = scope.tensor_use_owned(tensor, node_position);
        let rank = tensor.rank;
        let mut ranges = vec![quote! { .. }; rank];

        // Check if we have 1D tensor inputs
        let is_1d_start =
            matches!(&self.starts, SliceParam::Runtime(Type::Tensor(t)) if t.rank == 1);
        let is_1d_end = matches!(&self.ends, SliceParam::Runtime(Type::Tensor(t)) if t.rank == 1);

        if is_1d_start || is_1d_end {
            return self.generate_1d_tensor_slice(&input, scope, node_position, output, rank);
        }

        // Build slice ranges based on parameter types
        match (&self.starts, &self.ends) {
            // Both static: simple case
            (SliceParam::Static(starts), SliceParam::Static(ends)) => {
                let limit = starts.len().min(ends.len()).min(rank);
                for (i, range) in ranges.iter_mut().enumerate().take(limit) {
                    let start = starts[i].to_tokens();
                    let end = ends[i].to_tokens();
                    *range = quote! { #start..#end };
                }
            }

            // Both runtime shapes: multi-dimensional slicing
            (
                SliceParam::Runtime(Type::Shape(start_shape)),
                SliceParam::Runtime(Type::Shape(end_shape)),
            ) => {
                let start_name = &start_shape.name;
                let end_name = &end_shape.name;
                let num_dims = start_shape.rank.min(end_shape.rank).min(rank);

                for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    *range = quote! { #start_name[#idx]..#end_name[#idx] };
                }
            }

            // Static start, runtime shape end
            (SliceParam::Static(starts), SliceParam::Runtime(Type::Shape(end_shape))) => {
                let end_name = &end_shape.name;
                let num_dims = starts.len().min(end_shape.rank).min(rank);

                for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                    let start = starts[i].to_tokens();
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    *range = quote! { #start..#end_name[#idx] };
                }
            }

            // Runtime shape start, static end
            (SliceParam::Runtime(Type::Shape(start_shape)), SliceParam::Static(ends)) => {
                let start_name = &start_shape.name;
                let num_dims = start_shape.rank.min(ends.len()).min(rank);

                for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    let end = ends[i].to_tokens();
                    *range = quote! { #start_name[#idx]..#end };
                }
            }

            // Default: scalar slicing for first dimension
            _ => {
                let (start_expr, end_expr) = self.get_slice_range_expressions();
                ranges[0] = quote! { #start_expr..#end_expr };
            }
        }

        quote! {
            let #output = #input.slice(s![#(#ranges),*]);
        }
    }

    fn generate_1d_tensor_slice(
        &self,
        input: &TokenStream,
        scope: &mut Scope,
        node_position: usize,
        output: &proc_macro2::Ident,
        rank: usize,
    ) -> TokenStream {
        let mut ranges = vec![quote! { .. }; rank];

        // Get axes if specified
        let axes = if let Some(SliceParam::Static(axes)) = &self.axes {
            Some(axes.clone())
        } else {
            None
        };

        // Prepare start and end vectors
        let (start_setup, start_vec) = match &self.starts {
            SliceParam::Static(starts) => {
                let starts_lit = starts.iter().map(|&s| quote! { #s }).collect::<Vec<_>>();
                (quote! {}, quote! { alloc::vec![#(#starts_lit),*] })
            }
            SliceParam::Runtime(Type::Tensor(t)) if t.rank == 1 => {
                let tensor = scope.tensor_use_owned(t, node_position);
                (
                    quote! {
                        let start_data = #tensor.to_data();
                        let start_vec: alloc::vec::Vec<i64> = start_data.iter::<i64>().collect();
                    },
                    quote! { start_vec },
                )
            }
            _ => panic!("Invalid start parameter for 1D tensor slice"),
        };

        let (end_setup, end_vec) = match &self.ends {
            SliceParam::Static(ends) => {
                let ends_lit = ends.iter().map(|&e| quote! { #e }).collect::<Vec<_>>();
                (quote! {}, quote! { alloc::vec![#(#ends_lit),*] })
            }
            SliceParam::Runtime(Type::Tensor(t)) if t.rank == 1 => {
                let tensor = scope.tensor_use_owned(t, node_position);
                (
                    quote! {
                        let end_data = #tensor.to_data();
                        let end_vec: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    },
                    quote! { end_vec },
                )
            }
            _ => panic!("Invalid end parameter for 1D tensor slice"),
        };

        // Build ranges based on axes
        if let Some(axes_values) = axes {
            // We have specific axes to slice
            let mut slice_idx = 0;
            for i in 0..rank {
                let i_lit = proc_macro2::Literal::usize_unsuffixed(i);
                if axes_values.contains(&(i as i64)) {
                    let sidx = proc_macro2::Literal::usize_unsuffixed(slice_idx);

                    // Handle start based on type
                    let start_expr = match &self.starts {
                        SliceParam::Static(starts) if slice_idx < starts.len() => {
                            let val = starts[slice_idx];
                            quote! { #val as usize }
                        }
                        _ => quote! {
                            #start_vec.get(#sidx).map(|&s| s as usize).unwrap_or(0)
                        },
                    };

                    // Handle end based on type
                    let end_expr = match &self.ends {
                        SliceParam::Static(ends) if slice_idx < ends.len() => {
                            let val = ends[slice_idx];
                            quote! { #val as usize }
                        }
                        _ => quote! {
                            #end_vec.get(#sidx).map(|&e| e as usize).unwrap_or(input_dims[#i_lit])
                        },
                    };

                    ranges[i] = quote! { #start_expr..#end_expr };
                    slice_idx += 1;
                } else {
                    // This dimension is not sliced
                    ranges[i] = quote! { .. };
                }
            }
        } else {
            // No axes specified - slice dimensions based on vec lengths
            for i in 0..rank {
                let i_lit = proc_macro2::Literal::usize_unsuffixed(i);

                // Handle start based on type
                let start_expr = match &self.starts {
                    SliceParam::Static(starts) if i < starts.len() => {
                        let val = starts[i];
                        quote! { #val as usize }
                    }
                    _ => quote! {
                        #start_vec.get(#i_lit).map(|&s| s as usize).unwrap_or(0)
                    },
                };

                // Handle end based on type
                let end_expr = match &self.ends {
                    SliceParam::Static(ends) if i < ends.len() => {
                        let val = ends[i];
                        quote! { #val as usize }
                    }
                    _ => quote! {
                        #end_vec.get(#i_lit).map(|&e| e as usize).unwrap_or(input_dims[#i_lit])
                    },
                };

                ranges[i] = quote! { #start_expr..#end_expr };
            }
        }

        // Only include setup code if we have runtime parameters
        let setup = match (&self.starts, &self.ends) {
            (SliceParam::Static(_), SliceParam::Static(_)) => quote! {},
            (SliceParam::Static(_), SliceParam::Runtime(_)) => quote! { #end_setup },
            (SliceParam::Runtime(_), SliceParam::Static(_)) => quote! { #start_setup },
            (SliceParam::Runtime(_), SliceParam::Runtime(_)) => quote! { #start_setup #end_setup },
        };

        quote! {
            let input_dims = #input.dims();
            #setup
            let #output = #input.slice(s![#(#ranges),*]);
        }
    }

    fn get_slice_range_expressions(&self) -> (TokenStream, TokenStream) {
        let start_expr = match &self.starts {
            SliceParam::Static(starts) => starts[0].to_tokens(),
            SliceParam::Runtime(start_type) => self.get_scalar_expr(start_type),
        };

        let end_expr = match &self.ends {
            SliceParam::Static(ends) => ends[0].to_tokens(),
            SliceParam::Runtime(end_type) => self.get_scalar_expr(end_type),
        };

        (start_expr, end_expr)
    }

    fn generate_shape_slice(
        &self,
        shape: &crate::burn::ShapeType,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        let shape_name = &shape.name;
        let shape_len = Literal::usize_unsuffixed(shape.rank);

        // Get the output rank from the output type
        let output_rank = match &self.output {
            Type::Shape(output_shape) => output_shape.rank,
            _ => panic!("Expected Shape output type for shape slice operation"),
        };
        let output_rank_lit = Literal::usize_unsuffixed(output_rank);

        match (&self.starts, &self.ends) {
            (SliceParam::Static(starts), SliceParam::Static(ends)) if starts.len() == 1 => {
                let start_val = starts[0];
                let end_val = ends[0];

                // Handle special case: slice[-1:] pattern
                if start_val == -1 && (end_val == i64::MAX || end_val >= shape.rank as i64) {
                    // This gets the last element
                    quote! {
                        let #output: [i64; 1] = [#shape_name[#shape_name.len() - 1]];
                    }
                } else if start_val < 0 || end_val < 0 {
                    // Handle negative indices - convert at compile time since we know the shape length
                    let shape_len = shape.rank as i64;
                    let actual_start = if start_val < 0 {
                        (shape_len + start_val).max(0) as usize
                    } else {
                        start_val as usize
                    };
                    let actual_end = if end_val < 0 {
                        (shape_len + end_val).max(0) as usize
                    } else {
                        end_val as usize
                    };

                    let start_lit = Literal::usize_unsuffixed(actual_start);
                    let end_lit = Literal::usize_unsuffixed(actual_end);

                    quote! {
                        let #output: [i64; #output_rank_lit] = #shape_name[#start_lit..#end_lit].try_into().unwrap();
                    }
                } else {
                    // Positive indices
                    let start = start_val.to_tokens();
                    let end = if end_val == i64::MAX {
                        quote! { #shape_len }
                    } else {
                        end_val.to_tokens()
                    };
                    let output_len = if end_val == i64::MAX {
                        shape.rank.saturating_sub(start_val as usize)
                    } else {
                        (end_val as usize).saturating_sub(start_val as usize)
                    };
                    let output_rank = Literal::usize_unsuffixed(output_len);

                    quote! {
                        let #output: [i64; #output_rank] = #shape_name[s![#start..#end].into_ranges([#shape_len].into())[0].clone()].try_into().unwrap();
                    }
                }
            }
            _ => {
                // Runtime slicing - check if we have 1D tensor inputs
                match (&self.starts, &self.ends) {
                    (
                        SliceParam::Runtime(Type::Tensor(start_t)),
                        SliceParam::Runtime(Type::Tensor(end_t)),
                    ) if start_t.rank == 1 && end_t.rank == 1 => {
                        panic!(
                            "1D tensor slicing is not supported for shape inputs - shapes must be sliced with scalar or static indices"
                        );
                    }
                    _ => {
                        // Runtime slicing with scalars - we still know the output size from type inference
                        // and we know the shape length at compile time
                        let (start_expr, end_expr) = self.get_slice_range_expressions();
                        let shape_len_lit = Literal::i64_suffixed(shape.rank as i64);

                        quote! {
                            let _start_val = #start_expr as i64;
                            let _end_val = #end_expr as i64;
                            let _start = if _start_val < 0 { (#shape_len_lit + _start_val) as usize } else { _start_val as usize };
                            let _end = if _end_val < 0 { (#shape_len_lit + _end_val) as usize } else { _end_val as usize };
                            let #output: [i64; #output_rank_lit] = #shape_name[_start.._end].try_into().unwrap();
                        }
                    }
                }
            }
        }
    }

    fn get_scalar_expr(&self, scalar_type: &Type) -> TokenStream {
        match scalar_type {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            Type::Shape(shape) => {
                let name = &shape.name;
                // For single-dimension slicing, use the first element of the shape
                quote! { #name[0] }
            }
            Type::Tensor(tensor) if tensor.rank == 1 => {
                // For 1D tensor, we'll handle it specially in the calling code
                // This shouldn't be called for 1D tensors as they use different logic
                panic!(
                    "1D tensor slice parameters should be handled separately, not through get_scalar_expr"
                )
            }
            _ => panic!(
                "Expected scalar, shape, or 1D tensor type for runtime slice parameter, got {scalar_type:?}"
            ),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        let mut inputs = vec![self.input.clone()];

        // Add runtime inputs if needed
        if let SliceParam::Runtime(ref start_type) = self.starts {
            inputs.push(start_type.clone());
        }
        if let SliceParam::Runtime(ref end_type) = self.ends {
            inputs.push(end_type.clone());
        }

        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        self.generate_slice(scope, node_position)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::s");

        // Register Int if we have 1D tensor inputs
        if matches!(&self.starts, SliceParam::Runtime(Type::Tensor(t)) if t.rank == 1)
            || matches!(&self.ends, SliceParam::Runtime(Type::Tensor(t)) if t.rank == 1)
        {
            imports.register("burn::tensor::Int");
        }

        // For Shape slicing, we might need RangesArg
        if matches!(&self.input, Type::Shape(_)) {
            imports.register("burn::tensor::RangesArg");
        }
    }

    fn into_node(self) -> super::Node<PS> {
        super::Node::Slice(self)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{ShapeType, TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_slice_tensor_static() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
            SliceParam::Static(vec![0, 1, 2]),
            SliceParam::Static(vec![3, 4, 5]),
        ));
        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::s;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 3> {
                    let tensor2 = tensor1.slice(s![0..3, 1..4, 2..5]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_tensor_runtime_scalars() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "start",
                crate::burn::ScalarKind::Int64,
            ))),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "end",
                crate::burn::ScalarKind::Int64,
            ))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "start".to_string(),
                "end".to_string(),
            ],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 2>, start: i64, end: i64) -> Tensor<B, 2> {
                    let tensor2 = tensor1.slice(s![start..end, ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape_static() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 2)),
            SliceParam::Static(vec![1]),
            SliceParam::Static(vec![3]),
        ));
        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

        let expected = quote! {
            use burn::tensor::s;
            use burn::tensor::RangesArg;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, shape1: [i64; 4]) -> [i64; 2] {
                    let shape2: [i64; 2] = shape1[s![1..3].into_ranges([4].into())[0].clone()].try_into().unwrap();
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape_runtime() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 2)),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "start",
                crate::burn::ScalarKind::Int64,
            ))),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "end",
                crate::burn::ScalarKind::Int64,
            ))),
        ));
        graph.register_input_output(
            vec!["shape1".to_string(), "start".to_string(), "end".to_string()],
            vec!["shape2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
            use burn::tensor::RangesArg;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, shape1: [i64; 4], start: i64, end: i64) -> [i64; 2] {
                    let _start_val = start as i64;
                    let _end_val = end as i64;
                    let _start = if _start_val < 0 { (4i64 + _start_val) as usize } else { _start_val as usize };
                    let _end = if _end_val < 0 { (4i64 + _end_val) as usize } else { _end_val as usize };
                    let shape2: [i64; 2] = shape1[_start.._end].try_into().unwrap();
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_tensor_runtime_shapes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
            SliceParam::Runtime(Type::Shape(ShapeType::new("start_shape", 1))),
            SliceParam::Runtime(Type::Shape(ShapeType::new("end_shape", 1))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "start_shape".to_string(),
                "end_shape".to_string(),
            ],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 3>, start_shape: [i64; 1], end_shape: [i64; 1]) -> Tensor<B, 3> {
                    let tensor2 = tensor1.slice(s![start_shape[0]..end_shape[0], .., ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_1d_tensor_params() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
            SliceParam::Runtime(Type::Tensor(TensorType::new_int("starts", 1))),
            SliceParam::Runtime(Type::Tensor(TensorType::new_int("ends", 1))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "starts".to_string(),
                "ends".to_string(),
            ],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 3>, starts: Tensor<B, 1, Int>, ends: Tensor<B, 1, Int>) -> Tensor<B, 3> {
                    let input_dims = tensor1.dims();
                    let start_data = starts.to_data();
                    let start_vec: alloc::vec::Vec<i64> = start_data.iter::<i64>().collect();
                    let end_data = ends.to_data();
                    let end_vec: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    let tensor2 = tensor1.slice(s![
                        start_vec.get(0).map(|&s| s as usize).unwrap_or(0)..end_vec.get(0).map(|&e| e as usize).unwrap_or(input_dims[0]),
                        start_vec.get(1).map(|&s| s as usize).unwrap_or(0)..end_vec.get(1).map(|&e| e as usize).unwrap_or(input_dims[1]),
                        start_vec.get(2).map(|&s| s as usize).unwrap_or(0)..end_vec.get(2).map(|&e| e as usize).unwrap_or(input_dims[2])
                    ]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_mixed_1d_tensor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            SliceParam::Static(vec![0, 1]), // Static start
            SliceParam::Runtime(Type::Tensor(TensorType::new_int("ends", 1))), // 1D tensor end
        ));
        graph.register_input_output(
            vec!["tensor1".to_string(), "ends".to_string()],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 2>, ends: Tensor<B, 1, Int>) -> Tensor<B, 2> {
                    let input_dims = tensor1.dims();
                    let end_data = ends.to_data();
                    let end_vec: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    let tensor2 = tensor1.slice(s![
                        0i64 as usize..end_vec.get(0).map(|&e| e as usize).unwrap_or(input_dims[0]),
                        1i64 as usize..end_vec.get(1).map(|&e| e as usize).unwrap_or(input_dims[1])
                    ]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
