use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_log_softmax_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::log_softmax(tensor, 1);
    // log(softmax([1, 7])) and log(softmax([13, -3])); the large-positive row
    // exercises the detached max-shift (without it, exp(13) would overflow f16
    // and lose precision in f32).
    let expected = TensorData::from([[-6.0024757, -0.00247565], [0.0, -16.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
