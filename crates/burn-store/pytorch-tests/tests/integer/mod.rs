use burn::{
    module::{Module, Param, ParamId},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    buffer: Param<Tensor<B, 1, Int>>,
}

impl<B: Backend> Net<B> {
    /// Create a new model with placeholder values.
    pub fn init(device: &B::Device) -> Self {
        Self {
            buffer: Param::initialized(
                ParamId::new(),
                Tensor::<B, 1, Int>::from_data(TensorData::from([0, 0, 0]), device),
            ),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, _x: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        self.buffer.val()
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    fn integer(model: Net<TestBackend>) {
        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::ones([3, 3], &device);

        let output = model.forward(input);

        // Compare values without assuming a specific int dtype: .pt files
        // store i64, but the backend's native IntElem may be smaller.
        let values = output.to_data().iter::<i64>().collect::<Vec<_>>();
        assert_eq!(values, vec![1i64, 2, 3]);
    }

    #[test]
    fn integer_full_precision() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/integer/integer.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        integer(model);
    }

    #[test]
    fn integer_half_precision() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/integer/integer.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        integer(model);
    }
}
