#![allow(unexpected_cfgs)]
#[cfg(loom)]
mod loom_test {
    use loom::sync::{Arc, RwLock};
    use loom::thread;

    struct Coordinator {
        active_transactions: Arc<RwLock<()>>,
        commit_log: Arc<RwLock<()>>,
        connections: Arc<RwLock<()>>,
    }

    impl Coordinator {
        fn new() -> Self {
            Coordinator {
                active_transactions: Arc::new(RwLock::new(())),
                commit_log: Arc::new(RwLock::new(())),
                connections: Arc::new(RwLock::new(())),
            }
        }

        fn recover_pending_transactions(&self) {
            let _decisions = {
                let _log_guard = self.commit_log.read().unwrap();
                ()
            }; // explicitly drop the log guard just like in the real implementation

            // Reinserting recovered txns
            let _tx_guard = self.active_transactions.write().unwrap();

            // Getting connections to send commits
            let _conn_guard = self.connections.read().unwrap();
        }

        fn prepare_distributed_transaction(&self) {
            // First takes active_transactions write lock to remove tx
            {
                let _tx_guard = self.active_transactions.write().unwrap();
            }

            // Then connection read lock to send prepare requests
            let connections = self.connections.read().unwrap();

            drop(connections); // the fix

            let _tx_guard = self.active_transactions.write().unwrap();
        }

        fn commit_distributed_transaction(&self) {
            {
                let _tx_guard = self.active_transactions.write().unwrap();
            }

            {
                let _log_guard = self.commit_log.write().unwrap();
            } // FIX: Drop the commit log lock before taking the active transactions lock!

            // Modeling failure branch: reinsert_transaction
            let _tx_guard = self.active_transactions.write().unwrap();

            let connections = self.connections.read().unwrap();

            drop(connections); // the fix

            // reinsert_transaction
            let _tx_guard = self.active_transactions.write().unwrap();
        }
    }

    #[test]
    fn test_coordinator_deadlock() {
        loom::model(|| {
            let coord = Arc::new(Coordinator::new());

            let c1 = coord.clone();
            let t1 = thread::spawn(move || {
                c1.recover_pending_transactions();
            });

            let c2 = coord.clone();
            let t2 = thread::spawn(move || {
                c2.commit_distributed_transaction();
            });

            let c3 = coord.clone();
            let t3 = thread::spawn(move || {
                c3.prepare_distributed_transaction();
            });

            t1.join().unwrap();
            t2.join().unwrap();
            t3.join().unwrap();
        });
    }
}
