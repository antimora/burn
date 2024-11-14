use burn_router::{Runner, RunnerClient};
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{OperationDescription, ReprBackend, TensorDescription, TensorId},
    TensorData,
};
use core::marker::PhantomData;
use std::sync::mpsc::Sender;

use crate::shared::{ConnectionId, TaskResponse, TaskResponseContent};

/// The goal of the processor is to asynchronously process compute tasks on it own thread.
pub struct Processor<B: ReprBackend> {
    p: PhantomData<B>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(Box<OperationDescription>),
    RegisterTensor(TensorId, TensorData),
    ReadTensor(ConnectionId, TensorDescription, Callback<TaskResponse>),
    Sync(ConnectionId, Callback<TaskResponse>),
    Fence(Callback<()>),
    RegisterOrphan(TensorId),
    Close,
}

impl<B: ReprBackend> Processor<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn start(runner: Runner<B>) -> Sender<ProcessorTask> {
        let (sender, rec) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            for item in rec.iter() {
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        runner.register(*op);
                    }
                    ProcessorTask::RegisterOrphan(id) => {
                        runner.register_orphan(&id);
                    }
                    ProcessorTask::Sync(id, callback) => {
                        runner.sync();
                        callback
                            .send(TaskResponse {
                                content: TaskResponseContent::SyncBackend,
                                id,
                            })
                            .unwrap();
                    }
                    ProcessorTask::RegisterTensor(id, data) => {
                        runner.register_tensor_data_id(id, data);
                    }
                    ProcessorTask::ReadTensor(id, tensor, callback) => {
                        let tensor = burn_common::future::block_on(runner.read_tensor(tensor));
                        callback
                            .send(TaskResponse {
                                content: TaskResponseContent::ReadTensor(tensor),
                                id,
                            })
                            .unwrap();
                    }
                    ProcessorTask::Close => {
                        let device = runner.device();
                        runner.sync();
                        core::mem::drop(runner);
                        B::sync(&device);
                        return;
                    }
                    ProcessorTask::Fence(sender) => {
                        sender.send(()).unwrap();
                    }
                }
            }
        });

        sender
    }
}
