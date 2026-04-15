use super::*;
use burn_tensor::TensorData;

#[test]
fn expand_2d_bool() {
    let tensor = TestTensorBool::<1>::from([false, true, false]);
    let expanded_tensor = tensor.expand([3, 3]);

    let expected_data = TensorData::from([
        [false, true, false],
        [false, true, false],
        [false, true, false],
    ]);

    expanded_tensor.into_data().assert_eq(&expected_data, false);
}

#[test]
fn expand_bool_after_flip() {
    let tensor = TestTensorBool::<1>::from([true, false, true]).flip([0]);

    let output = tensor.expand([2, 3]);

    output.into_data().assert_eq(
        &TensorData::from([[true, false, true], [true, false, true]]),
        false,
    );
}
