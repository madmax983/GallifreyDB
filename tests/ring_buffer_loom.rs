#[cfg(loom)]
mod loom_tests {
    use loom::thread;
    use loom::sync::Arc;
    use aletheiadb::storage::wal::ring_buffer::{WalRingBuffer, PendingEntry};
    use aletheiadb::storage::LSN;

    #[test]
    fn test_havoc_loom_concurrency() {
        loom::model(|| {
            let buffer = Arc::new(WalRingBuffer::new(2));

            let b1 = buffer.clone();
            let b2 = buffer.clone();

            let t1 = thread::spawn(move || {
                let entry = PendingEntry::new_async(LSN(1), vec![1]);
                let _ = b1.try_append(entry);
            });

            let t2 = thread::spawn(move || {
                let entry = PendingEntry::new_async(LSN(2), vec![2]);
                let _ = b2.try_append(entry);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let drained = buffer.drain();
            assert!(drained.len() <= 2);
        });
    }

    #[test]
    fn test_havoc_loom_wrap_around() {
        loom::model(|| {
            let buffer = Arc::new(WalRingBuffer::new(2));

            let b1 = buffer.clone();
            let b2 = buffer.clone();
            let b3 = buffer.clone();
            let b4 = buffer.clone();

            let t1 = thread::spawn(move || {
                let _ = b1.try_append(PendingEntry::new_async(LSN(1), vec![1]));
            });

            let t2 = thread::spawn(move || {
                let _ = b2.try_append(PendingEntry::new_async(LSN(2), vec![2]));
            });

            let t3 = thread::spawn(move || {
                let _ = b3.try_append(PendingEntry::new_async(LSN(3), vec![3]));
            });

            let t4 = thread::spawn(move || {
                let _ = b4.drain();
            });

            t1.join().unwrap();
            t2.join().unwrap();
            t3.join().unwrap();
            t4.join().unwrap();

            let _ = buffer.drain();
        });
    }

    #[test]
    fn test_havoc_loom_backpressure() {
        loom::model(|| {
            let buffer = Arc::new(WalRingBuffer::new(2));

            let b1 = buffer.clone();
            let b2 = buffer.clone();
            let b3 = buffer.clone();
            let b_drain = buffer.clone();

            let t1 = thread::spawn(move || {
                let _ = b1.try_append(PendingEntry::new_async(LSN(1), vec![1]));
                let _ = b1.try_append(PendingEntry::new_async(LSN(4), vec![4]));
            });

            let t2 = thread::spawn(move || {
                let _ = b2.try_append(PendingEntry::new_async(LSN(2), vec![2]));
                let _ = b2.try_append(PendingEntry::new_async(LSN(5), vec![5]));
            });

            let t3 = thread::spawn(move || {
                let _ = b3.try_append(PendingEntry::new_async(LSN(3), vec![3]));
                let _ = b3.try_append(PendingEntry::new_async(LSN(6), vec![6]));
            });

            let t_drain = thread::spawn(move || {
                let _ = b_drain.drain();
                let _ = b_drain.drain();
                let _ = b_drain.drain();
            });

            t1.join().unwrap();
            t2.join().unwrap();
            t3.join().unwrap();
            t_drain.join().unwrap();

            let _ = buffer.drain();
        });
    }
}
