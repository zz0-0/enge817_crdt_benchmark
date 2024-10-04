use std::{
    cell::RefCell,
    collections::HashSet,
    fs::OpenOptions,
    sync::{Arc, Mutex, RwLock},
    thread,
    time::Duration,
};

use crdts::{CmRDT, CvRDT, GCounter, GList, GSet, LWWReg, List, MVReg, Map, Orswot, PNCounter};
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use memory_stats::memory_stats;
use rand::Rng;
use std::io::Write;

type TestActor = u32;
type TestKey = u32;
type TestVal = MVReg<u32, TestActor>;
type TestMap = Map<TestKey, TestVal, TestActor>;
const NUM_THREADS: usize = 4;
struct MemoryUsage {
    physical: usize,
    virtual_mem: usize,
}
#[derive(Default)]
struct MemoryStats {
    total_physical: usize,
    total_virtual: usize,
    iterations: usize,
}

fn measure_memory() -> Option<MemoryUsage> {
    memory_stats().map(|usage| MemoryUsage {
        physical: usage.physical_mem,
        virtual_mem: usage.virtual_mem,
    })
}

fn custom_benchmark<F>(c: &mut BenchmarkGroup<'_, WallTime>, id: &str, mut setup: F)
where
    F: FnMut() -> Box<dyn FnMut()>,
{
    let mut stats = MemoryStats::default();

    c.bench_function(id, |b| {
        let mut inner_fn = setup();
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let start_mem = measure_memory();
                inner_fn();
                let end_mem = measure_memory();

                if let (Some(start), Some(end)) = (start_mem, end_mem) {
                    stats.total_physical += end.physical.saturating_sub(start.physical);
                    stats.total_virtual += end.virtual_mem.saturating_sub(start.virtual_mem);
                }
                stats.iterations += 1;
            }
            start.elapsed()
        });
    });

    let avg_physical = stats.total_physical as f64 / stats.iterations as f64;
    let avg_virtual = stats.total_virtual as f64 / stats.iterations as f64;
    // println!(
    //     "{} Avg Memory Usage: Physical: {:.2} bytes, Virtual: {:.2} bytes",
    //     id, avg_physical, avg_virtual
    // );

    let result = format!(
        "{} Avg Memory Usage: Physical: {:.2} bytes, Virtual: {:.2} bytes",
        id, avg_physical, avg_virtual
    );

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("benchmark_results.txt")
        .expect("Failed to open file");

    writeln!(file, "{}", result).expect("Failed to write to file");
}

fn insert_single_benchmark(c: &mut Criterion) {
    // Create a benchmark group for Insert Operations
    // let mut group = c.benchmark_group("Insert Single Operations");
    let mut group = c.benchmark_group("Insert Single Operations");

    // Initialize CRDTs
    let gset = GSet::new();
    let mut orswot = Orswot::new();
    let gcounter = GCounter::new();
    let pncounter = PNCounter::new();
    let lww_reg = LWWReg { val: 42, marker: 0 };
    let mut mv_reg = MVReg::new();
    let mut map: TestMap = Map::new();
    let mut list = List::new();
    let mut glist = GList::new();

    custom_benchmark(&mut group, "GSet Insert", || {
        Box::new({
            let mut value = gset.clone();
            move || {
                value.insert(black_box("element"));
            }
        })
    });

    custom_benchmark(&mut group, "Orswot Insert", || {
        Box::new({
            let mut value = orswot.clone();
            move || {
                let add_ctx = value.read_ctx().derive_add_ctx(1);
                let add_op = value.add("value", add_ctx);
                value.apply(add_op);
            }
        })
    });

    custom_benchmark(&mut group, "GCounter Increment", || {
        Box::new({
            let value = gcounter.clone();
            move || {
                value.inc(black_box(1));
            }
        })
    });

    custom_benchmark(&mut group, "PNCounter Increment", || {
        Box::new({
            let value = pncounter.clone();
            move || {
                value.inc(black_box(1));
            }
        })
    });

    custom_benchmark(&mut group, "LWWReg Set", || {
        Box::new({
            let mut value = lww_reg.clone();
            move || {
                value.update(black_box(1), black_box(100));
            }
        })
    });

    custom_benchmark(&mut group, "MVReg Set", || {
        Box::new({
            let mut value = mv_reg.clone();
            move || {
                let read_ctx = value.read_ctx();
                let add_ctx = read_ctx.derive_add_ctx(1);
                let op = value.write(black_box("new_value"), add_ctx);
                value.apply(op);
            }
        })
    });

    custom_benchmark(&mut group, "Map Insert", || {
        Box::new({
            let mut value = map.clone();
            move || {
                let actor = 1;
                let add_ctx = value.read_ctx().derive_add_ctx(actor);
                let op = value.update(black_box(1 as u32), add_ctx, |v, ctx| {
                    v.write(black_box(42), ctx)
                });
                value.apply(op);
            }
        })
    });

    custom_benchmark(&mut group, "List Insert", || {
        Box::new({
            let mut value = list.clone();
            move || {
                let op = value.insert_index(black_box(0), black_box("element"), black_box("actor"));
                value.apply(op);
            }
        })
    });

    custom_benchmark(&mut group, "GList Insert", || {
        Box::new({
            let mut value = glist.clone();
            move || {
                let op = value.insert_after(None, black_box("element"));
                value.apply(op);
            }
        })
    });

    // Finish the group benchmark
    group.finish();
}

fn insert_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Insert Multiple Operations");

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        custom_benchmark(
            &mut group,
            &format!("GSet Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let mut gset = GSet::new();
                    for i in 0..*size {
                        gset.insert(black_box(i));
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Orswot Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let mut orswot = Orswot::new();
                    for i in 0..*size {
                        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                        let add_op = orswot.add(black_box(i), add_ctx);
                        orswot.apply(add_op);
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GCounter Multiple Increment {}", size),
            || {
                Box::new(move || {
                    let gcounter = GCounter::new();
                    for _ in 0..*size {
                        gcounter.inc(black_box(1));
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("PNCounter Multiple Increment {}", size),
            || {
                Box::new(move || {
                    let pncounter = PNCounter::new();
                    for _ in 0..*size {
                        pncounter.inc(black_box(1));
                    }
                })
            },
        );

        custom_benchmark(&mut group, &format!("LWWReg Multiple Set {}", size), || {
            Box::new(move || {
                let mut lww_reg = LWWReg { val: 0, marker: 0 };
                for i in 0..*size {
                    lww_reg.update(black_box(i as u64), black_box(i));
                }
            })
        });

        custom_benchmark(&mut group, &format!("MVReg Multiple Set {}", size), || {
            Box::new(move || {
                let mut mv_reg = MVReg::new();
                for i in 0..*size {
                    let read_ctx = mv_reg.read_ctx();
                    let add_ctx = read_ctx.derive_add_ctx(1);
                    let op = mv_reg.write(black_box(i), add_ctx);
                    mv_reg.apply(op);
                }
            })
        });

        custom_benchmark(&mut group, &format!("Map Multiple Insert {}", size), || {
            Box::new(move || {
                let mut map: TestMap = Map::new();
                for i in 0..*size {
                    let actor = 1;
                    let add_ctx = map.read_ctx().derive_add_ctx(actor);
                    let op = map.update(black_box(i as u32), add_ctx, |v, ctx| {
                        v.write(black_box(i as u32), ctx)
                    });
                    map.apply(op);
                }
            })
        });

        custom_benchmark(
            &mut group,
            &format!("List Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let mut list = List::new();
                    for i in 0..*size {
                        let op = list.insert_index(black_box(i), black_box(i), black_box("actor"));
                        list.apply(op);
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GList Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let mut glist = GList::new();
                    for i in 0..*size {
                        let op = glist.insert_after(None, black_box(i));
                        glist.apply(op);
                    }
                })
            },
        );
    }

    group.finish();
}

fn concurrent_insert_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Insert Operations");

    // Initialize CRDTs wrapped in Arc<Mutex<>>
    let gset = Arc::new(Mutex::new(GSet::new()));
    let orswot = Arc::new(Mutex::new(Orswot::new()));
    let gcounter = Arc::new(Mutex::new(GCounter::new()));
    let pncounter = Arc::new(Mutex::new(PNCounter::new()));
    let lww_reg = Arc::new(Mutex::new(LWWReg { val: 42, marker: 0 }));
    let mv_reg = Arc::new(Mutex::new(MVReg::new()));
    let map: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));
    let list = Arc::new(Mutex::new(List::new()));
    let glist = Arc::new(Mutex::new(GList::new()));

    // Benchmark for Concurrent GSet Insert
    custom_benchmark(&mut group, "Concurrent GSet Insert", || {
        let gset = Arc::clone(&gset);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = gset.clone();
                        move || {
                            let mut gset = value.lock().unwrap();
                            gset.insert(black_box(format!("element{}", i)));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent ORSWOT Insert
    custom_benchmark(&mut group, "Concurrent Orswot Insert", || {
        let orswot = Arc::clone(&orswot);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = orswot.clone();
                        move || {
                            let mut orswot = value.lock().unwrap();
                            let add_ctx = orswot.read_ctx().derive_add_ctx(i as u64);
                            let add_op = orswot.add(black_box(format!("value{}", i)), add_ctx);
                            orswot.apply(add_op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent GCounter Increment
    custom_benchmark(&mut group, "Concurrent GCounter Increment", || {
        let gcounter = Arc::clone(&gcounter);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|_| {
                    thread::spawn({
                        let value = gcounter.clone();
                        move || {
                            let mut gcounter = value.lock().unwrap();
                            gcounter.inc(black_box(1));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent PNCounter Increment
    custom_benchmark(&mut group, "Concurrent PNCounter Increment", || {
        let pncounter = Arc::clone(&pncounter);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|_| {
                    thread::spawn({
                        let value = pncounter.clone();
                        move || {
                            let mut pncounter = value.lock().unwrap();
                            pncounter.inc(black_box(1));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent LWWReg Set
    custom_benchmark(&mut group, "Concurrent LWWReg Set", || {
        let lww_reg = Arc::clone(&lww_reg);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = lww_reg.clone();
                        move || {
                            let mut lww_reg = value.lock().unwrap();
                            lww_reg.update(black_box(i as u64), black_box(100 + i));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent MVReg Set
    custom_benchmark(&mut group, "Concurrent MVReg Set", || {
        let mv_reg = Arc::clone(&mv_reg);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = mv_reg.clone();
                        move || {
                            let mut mv_reg = value.lock().unwrap();
                            let read_ctx = mv_reg.read_ctx();
                            let add_ctx = read_ctx.derive_add_ctx(i as u64);
                            let op = mv_reg.write(black_box(format!("new_value{}", i)), add_ctx);
                            mv_reg.apply(op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent Map Insert/Update
    custom_benchmark(&mut group, "Concurrent Map Insert", || {
        let map = Arc::clone(&map);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = map.clone();
                        move || {
                            let mut map = value.lock().unwrap();
                            let actor = i;
                            let add_ctx = map.read_ctx().derive_add_ctx(actor.try_into().unwrap());
                            let op = map.update(black_box(i as u32), add_ctx, |v, ctx| {
                                v.write(black_box((42 + i).try_into().unwrap()), ctx)
                            });
                            map.apply(op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent List Insert
    custom_benchmark(&mut group, "Concurrent List Insert", || {
        let list = Arc::clone(&list);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = list.clone();
                        move || {
                            let mut list = value.lock().unwrap();
                            let op = list.insert_index(
                                black_box(0),
                                black_box(format!("element{}", i)),
                                black_box(format!("actor{}", i)),
                            );
                            list.apply(op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent GList Insert
    custom_benchmark(&mut group, "Concurrent GList Insert", || {
        let glist = Arc::clone(&glist);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value = glist.clone();
                        move || {
                            let mut glist = value.lock().unwrap();
                            let op = glist.insert_after(None, black_box(format!("element{}", i)));
                            glist.apply(op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    group.finish();
}

fn concurrent_insert_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Insert Multiple Operations");

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in sizes.iter() {
        let inserts_per_thread = size / NUM_THREADS;

        custom_benchmark(
            &mut group,
            &format!("Concurrent GSet Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let gset = Arc::new(Mutex::new(GSet::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let gset = Arc::clone(&gset);
                            thread::spawn(move || {
                                let mut gset = gset.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    gset.insert(black_box(thread_id * inserts_per_thread + i));
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent Orswot Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let orswot = Arc::new(Mutex::new(Orswot::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let orswot = Arc::clone(&orswot);
                            thread::spawn(move || {
                                let mut orswot = orswot.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    let add_ctx =
                                        orswot.read_ctx().derive_add_ctx(thread_id as u64);
                                    let add_op = orswot.add(
                                        black_box(thread_id * inserts_per_thread + i),
                                        add_ctx,
                                    );
                                    orswot.apply(add_op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent GCounter Multiple Increment {}", size),
            || {
                Box::new(move || {
                    let gcounter = Arc::new(Mutex::new(GCounter::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            let gcounter = Arc::clone(&gcounter);
                            thread::spawn(move || {
                                let mut gcounter = gcounter.lock().unwrap();
                                for _ in 0..inserts_per_thread {
                                    gcounter.inc(black_box(1));
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent PNCounter Multiple Increment {}", size),
            || {
                Box::new(move || {
                    let pncounter = Arc::new(Mutex::new(PNCounter::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            let pncounter = Arc::clone(&pncounter);
                            thread::spawn(move || {
                                let mut pncounter = pncounter.lock().unwrap();
                                for _ in 0..inserts_per_thread {
                                    pncounter.inc(black_box(1));
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent LWWReg Multiple Set {}", size),
            || {
                Box::new(move || {
                    let lww_reg = Arc::new(Mutex::new(LWWReg { val: 0, marker: 0 }));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let lww_reg = Arc::clone(&lww_reg);
                            thread::spawn(move || {
                                let mut lww_reg = lww_reg.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    lww_reg.update(
                                        black_box((thread_id * inserts_per_thread + i) as u64),
                                        black_box(i),
                                    );
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent MVReg Multiple Set {}", size),
            || {
                Box::new(move || {
                    let mv_reg = Arc::new(Mutex::new(MVReg::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let mv_reg = Arc::clone(&mv_reg);
                            thread::spawn(move || {
                                let mut mv_reg = mv_reg.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    let read_ctx = mv_reg.read_ctx();
                                    let add_ctx = read_ctx.derive_add_ctx(thread_id as u64);
                                    let op = mv_reg.write(
                                        black_box(thread_id * inserts_per_thread + i),
                                        add_ctx,
                                    );
                                    mv_reg.apply(op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent Map Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let map: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let map = Arc::clone(&map);
                            thread::spawn(move || {
                                let mut map = map.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    let actor = thread_id as u64;
                                    let add_ctx =
                                        map.read_ctx().derive_add_ctx(actor.try_into().unwrap());
                                    let op = map.update(
                                        black_box((thread_id * inserts_per_thread + i) as u32),
                                        add_ctx,
                                        |v, ctx| v.write(black_box(i as u32), ctx),
                                    );
                                    map.apply(op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent List Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let list = Arc::new(Mutex::new(List::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let list = Arc::clone(&list);
                            thread::spawn(move || {
                                let mut list = list.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    let op = list.insert_index(
                                        black_box(thread_id * inserts_per_thread + i),
                                        black_box(i),
                                        black_box(format!("actor{}", thread_id)),
                                    );
                                    list.apply(op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent GList Multiple Insert {}", size),
            || {
                Box::new(move || {
                    let glist = Arc::new(Mutex::new(GList::new()));
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let glist = Arc::clone(&glist);
                            thread::spawn(move || {
                                let mut glist = glist.lock().unwrap();
                                for i in 0..inserts_per_thread {
                                    let op = glist.insert_after(
                                        None,
                                        black_box(thread_id * inserts_per_thread + i),
                                    );
                                    glist.apply(op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

fn update_single_benchmark(c: &mut Criterion) {
    // Create a benchmark group for Update Operations
    let mut group = c.benchmark_group("Update Operations");

    // Initialize CRDTs
    let lww_reg = LWWReg { val: 42, marker: 0 };
    let mv_reg = MVReg::new();
    let mut map: TestMap = Map::new();
    let mut orswot = Orswot::new();
    let mut list = List::new();

    // Pre-insert values for update to work
    let actor = 1;
    let add_ctx = map.read_ctx().derive_add_ctx(actor);
    let op = map.update(black_box(1 as u32), add_ctx, |v, ctx| {
        v.write(black_box(42), ctx)
    });
    map.apply(op);

    let add_ctx = orswot.read_ctx().derive_add_ctx(1);
    let add_op = orswot.add("value", add_ctx);
    orswot.apply(add_op);

    let op = list.insert_index(black_box(0), black_box("element"), black_box("actor"));
    list.apply(op);

    custom_benchmark(&mut group, "LWWReg Update", || {
        Box::new({
            let value = lww_reg.clone();
            move || {
                let mut value = value.clone();
                value.update(black_box(100), black_box(1));
            }
        })
    });

    custom_benchmark(&mut group, "MVReg Update", || {
        Box::new({
            let value = mv_reg.clone();
            move || {
                let mut value = value.clone();
                let ctx = value.read_ctx().derive_add_ctx(1);
                let op = value.write(black_box("new_value"), ctx);
                value.apply(op);
            }
        })
    });

    custom_benchmark(&mut group, "Map Update", || {
        Box::new({
            let value = map.clone();
            move || {
                let mut value = value.clone();
                let actor = 1;
                let add_ctx = value.read_ctx().derive_add_ctx(actor);
                let op = value.update(black_box(1 as u32), add_ctx, |v, ctx| {
                    v.write(black_box(84), ctx)
                });
                value.apply(op);
            }
        })
    });

    custom_benchmark(&mut group, "Orswot Update (Re-insert)", || {
        Box::new({
            let value = orswot.clone();
            move || {
                let mut value = value.clone();
                let ctx = value.read_ctx().derive_add_ctx(1);
                let op = value.add(black_box("updated_value"), ctx);
                value.apply(op);
            }
        })
    });

    // Finish the group benchmark
    group.finish();
}

fn update_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Update Multiple Operations");

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        custom_benchmark(
            &mut group,
            &format!("LWWReg Multiple Update {}", size),
            || {
                Box::new(move || {
                    let mut lww_reg = LWWReg { val: 0, marker: 0 };
                    for i in 0..*size {
                        lww_reg.update(black_box(i as u64), black_box(i));
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("MVReg Multiple Update {}", size),
            || {
                Box::new(move || {
                    let mut mv_reg = MVReg::new();
                    for i in 0..*size {
                        let ctx = mv_reg.read_ctx().derive_add_ctx(1);
                        let op = mv_reg.write(black_box(format!("value_{}", i)), ctx);
                        mv_reg.apply(op);
                    }
                })
            },
        );

        custom_benchmark(&mut group, &format!("Map Multiple Update {}", size), || {
            Box::new(move || {
                let mut map: TestMap = Map::new();
                for i in 0..*size {
                    let actor = 1;
                    let add_ctx = map.read_ctx().derive_add_ctx(actor);
                    let op = map.update(black_box(i as u32), add_ctx, |v, ctx| {
                        v.write(black_box(i * 2), ctx)
                    });
                    map.apply(op);
                }
            })
        });

        custom_benchmark(
            &mut group,
            &format!("Orswot Multiple Update (Re-insert) {}", size),
            || {
                Box::new(move || {
                    let mut orswot = Orswot::new();
                    for i in 0..*size {
                        let ctx = orswot.read_ctx().derive_add_ctx(1);
                        let op = orswot.add(black_box(format!("value_{}", i)), ctx);
                        orswot.apply(op);
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("List Multiple Update (Replace) {}", size),
            || {
                Box::new(move || {
                    let mut list = List::new();
                    // First, insert initial elements
                    for i in 0..*size {
                        let op = list.insert_index(
                            (i as u32).try_into().unwrap(),
                            format!("element_{}", i),
                            "actor",
                        );
                        list.apply(op);
                    }
                    // Then, update (replace) all elements
                    for i in 0..*size {
                        if let Some(del_op) =
                            list.delete_index((i as u32).try_into().unwrap(), "actor")
                        {
                            list.apply(del_op);
                        }
                        let ins_op = list.insert_index(
                            (i as u32).try_into().unwrap(),
                            format!("updated_element_{}", i),
                            "actor",
                        );
                        list.apply(ins_op);
                    }
                })
            },
        );
    }

    group.finish();
}

fn concurrent_update_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Update Single Operations");
    const NUM_THREADS: usize = 4;

    // Initialize CRDTs wrapped in Arc<Mutex<>>
    let lww_reg = Arc::new(Mutex::new(LWWReg { val: 42, marker: 0 }));
    let mv_reg = Arc::new(Mutex::new(MVReg::new()));
    let map: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));
    let orswot = Arc::new(Mutex::new(Orswot::new()));
    let list = Arc::new(Mutex::new(List::new()));

    // Pre-insert values for update to work
    {
        let mut map = map.lock().unwrap();
        let actor = 1;
        let add_ctx = map.read_ctx().derive_add_ctx(actor);
        let op = map.update(black_box(1 as u32), add_ctx, |v, ctx| {
            v.write(black_box(42), ctx)
        });
        map.apply(op);
    }
    {
        let mut orswot = orswot.lock().unwrap();
        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
        let add_op = orswot.add("value", add_ctx);
        orswot.apply(add_op);
    }
    {
        let mut list = list.lock().unwrap();
        let op = list.insert_index(black_box(0), black_box("element"), black_box("actor"));
        list.apply(op);
    }

    // Benchmark for Concurrent LWWReg Update
    custom_benchmark(&mut group, "Concurrent LWWReg Update", || {
        Box::new({
            let value = lww_reg.clone();
            move || {
                let lww_reg = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|thread_id| {
                        thread::spawn({
                            let value = lww_reg.clone();
                            move || {
                                let mut lww_reg = value.lock().unwrap();
                                lww_reg.update(
                                    black_box(100 + thread_id as u64),
                                    black_box(thread_id as u64),
                                );
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent MVReg Update
    custom_benchmark(&mut group, "Concurrent MVReg Update", || {
        Box::new({
            let value = mv_reg.clone();
            move || {
                let mv_reg = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|thread_id| {
                        thread::spawn({
                            let value = mv_reg.clone();
                            move || {
                                let mut mv_reg = value.lock().unwrap();
                                let ctx = mv_reg.read_ctx().derive_add_ctx(thread_id as u64);
                                mv_reg.write(black_box(format!("new_value_{}", thread_id)), ctx);
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent Map Update
    custom_benchmark(&mut group, "Concurrent Map Update", || {
        Box::new({
            let value = map.clone();
            move || {
                let map = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|thread_id| {
                        thread::spawn({
                            let value = map.clone();
                            move || {
                                let mut map = value.lock().unwrap();
                                let actor = thread_id as u64;
                                let add_ctx =
                                    map.read_ctx().derive_add_ctx(actor.try_into().unwrap());
                                let op = map.update(black_box(1 as u32), add_ctx, |v, ctx| {
                                    v.write(black_box((84 + thread_id).try_into().unwrap()), ctx)
                                });
                                map.apply(op);
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    custom_benchmark(&mut group, "Concurrent Orswot Update", || {
        Box::new({
            let value = orswot.clone();
            move || {
                let orswot = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|thread_id| {
                        thread::spawn({
                            let value = orswot.clone();
                            move || {
                                let mut orswot = value.lock().unwrap();
                                let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                                let rm_op = orswot.rm("value", rm_ctx);
                                orswot.apply(rm_op);

                                let add_ctx = orswot.read_ctx().derive_add_ctx(thread_id as u64);
                                let add_op = orswot.add("new value", add_ctx);
                                orswot.apply(add_op);
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    group.finish();
}

fn concurrent_update_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Update Operations");
    const NUM_THREADS: usize = 4;
    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        custom_benchmark(
            &mut group,
            &format!("GSet Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let gset = Arc::new(RwLock::new(GSet::new()));
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut rng = rand::thread_rng();
                                for _ in 0..size / NUM_THREADS {
                                    let mut gset = gset.write().unwrap();
                                    gset.insert(rng.gen::<u32>());
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Orswot Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let orswot = Arc::new(RwLock::new(Orswot::new()));
                    thread::scope(|s| {
                        for thread_id in 0..NUM_THREADS {
                            let orswot = Arc::clone(&orswot);
                            s.spawn(move || {
                                let mut rng = rand::thread_rng();
                                for _ in 0..size / NUM_THREADS {
                                    let mut orswot = orswot.write().unwrap();
                                    let add_ctx =
                                        orswot.read_ctx().derive_add_ctx(thread_id as u64);
                                    let op = orswot.add(rng.gen::<u32>(), add_ctx);
                                    orswot.apply(op);
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GCounter Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let gcounter = Arc::new(RwLock::new(GCounter::new()));
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut rng = rand::thread_rng();
                                for _ in 0..size / NUM_THREADS {
                                    let mut gcounter = gcounter.write().unwrap();
                                    let op = gcounter.inc(rng.gen_range(1..10));
                                    gcounter.apply(op);
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("PNCounter Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let pncounter = Arc::new(RwLock::new(PNCounter::new()));
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut rng = rand::thread_rng();
                                for _ in 0..size / NUM_THREADS {
                                    let mut pncounter = pncounter.write().unwrap();
                                    if rng.gen_bool(0.5) {
                                        let op = pncounter.inc(rng.gen_range(1..10));
                                        pncounter.apply(op);
                                    } else {
                                        let op = pncounter.dec(rng.gen_range(1..10));
                                        pncounter.apply(op);
                                    }
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("LWWReg Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let lwwreg = Arc::new(RwLock::new(LWWReg { val: 0, marker: 0 }));
                    thread::scope(|s| {
                        for thread_id in 0..NUM_THREADS {
                            let lwwreg = Arc::clone(&lwwreg);
                            s.spawn(move || {
                                let mut rng = rand::thread_rng();
                                for i in 0..size / NUM_THREADS {
                                    let mut lwwreg = lwwreg.write().unwrap();
                                    lwwreg.update(
                                        (thread_id * size / NUM_THREADS + i) as u64,
                                        rng.gen::<u32>(),
                                    );
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("MVReg Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let mvreg = Arc::new(RwLock::new(MVReg::new()));
                    thread::scope(|s| {
                        for thread_id in 0..NUM_THREADS {
                            let mvreg = Arc::clone(&mvreg);
                            s.spawn(move || {
                                let mut rng = rand::thread_rng();
                                for _ in 0..size / NUM_THREADS {
                                    let mut mvreg = mvreg.write().unwrap();
                                    let ctx = mvreg.read_ctx().derive_add_ctx(thread_id as u64);
                                    let op = mvreg.write(rng.gen::<u32>(), ctx);
                                    mvreg.apply(op);
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Map Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let map: Arc<RwLock<TestMap>> = Arc::new(RwLock::new(Map::new()));
                    thread::scope(|s| {
                        for thread_id in 0..NUM_THREADS {
                            let map = Arc::clone(&map);
                            s.spawn(move || {
                                let mut rng = rand::thread_rng();
                                for i in 0..size / NUM_THREADS {
                                    let mut map = map.write().unwrap();
                                    let add_ctx = map
                                        .read_ctx()
                                        .derive_add_ctx((thread_id as u64).try_into().unwrap());
                                    let op = map.update(i as u32, add_ctx, |v, ctx| {
                                        v.write(rng.gen::<u32>().try_into().unwrap(), ctx)
                                    });
                                    map.apply(op);
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("List Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let list = Arc::new(RwLock::new(List::new()));
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut rng = rand::thread_rng();
                                for i in 0..size / NUM_THREADS {
                                    let mut list = list.write().unwrap();
                                    let op = list.insert_index(i, rng.gen::<u32>(), "actor");
                                    list.apply(op);
                                }
                            });
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GList Concurrent Update {}", size),
            || {
                Box::new(move || {
                    let glist = Arc::new(RwLock::new(GList::new()));
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut rng = rand::thread_rng();
                                for _ in 0..size / NUM_THREADS {
                                    let mut glist = glist.write().unwrap();
                                    let op = glist.insert_after(glist.last(), rng.gen::<u32>());
                                    glist.apply(op);
                                }
                            });
                        }
                    });
                })
            },
        );
    }

    group.finish();
}

fn remove_single_benchmark(c: &mut Criterion) {
    // Create a benchmark group for Remove Operations
    let mut group = c.benchmark_group("Remove Operations");

    // Initialize CRDTs
    let mut orswot = Orswot::new();
    let pncounter = PNCounter::new();
    let mut map: TestMap = Map::new();
    let mut list = List::new();

    // Pre-insert values for removal to work
    let add_ctx = orswot.read_ctx().derive_add_ctx(1);
    let add_op = orswot.add("value", add_ctx);
    orswot.apply(add_op);
    let actor = 1;
    let add_ctx = map.read_ctx().derive_add_ctx(actor);
    let op = map.update(black_box(1 as u32), add_ctx, |v, ctx| {
        v.write(black_box(42), ctx)
    });
    map.apply(op);
    pncounter.inc(1);
    let op = list.insert_index(black_box(0), black_box("element"), black_box("actor"));
    list.apply(op);

    // Benchmark for ORSWOT Remove
    custom_benchmark(&mut group, "Orswot Remove", || {
        Box::new({
            let mut value = orswot.clone();
            move || {
                let rm_ctx = value.read_ctx().derive_rm_ctx();
                let rm_op = value.rm(black_box("value"), rm_ctx);
                value.apply(rm_op);
            }
        })
    });

    // Benchmark for PNCounter Decrement
    custom_benchmark(&mut group, "PNCounter Decrement", || {
        Box::new({
            let mut value = pncounter.clone();
            move || {
                value.dec(black_box(1));
            }
        })
    });

    // Benchmark for Map Remove
    custom_benchmark(&mut group, "Map Remove", || {
        Box::new({
            let mut value = map.clone();
            move || {
                let rm_ctx = value.read_ctx().derive_rm_ctx();
                let rm_op = value.rm(black_box(1 as u32), rm_ctx);
                value.apply(rm_op);
            }
        })
    });

    // Benchmark for List Remove
    custom_benchmark(&mut group, "List Remove", || {
        Box::new({
            let mut value = list.clone();
            move || {
                if let Some(del_op) = value.delete_index(black_box(0), black_box("actor")) {
                    value.apply(del_op);
                }
            }
        })
    });

    // Finish the group benchmark
    group.finish();
}

fn remove_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Remove Multiple Operations");

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        // Orswot Remove
        custom_benchmark(
            &mut group,
            &format!("Orswot Multiple Remove {}", size),
            || {
                Box::new(move || {
                    let mut orswot = Orswot::new();
                    for i in 0..*size {
                        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                        let op = orswot.add(format!("value_{}", i), add_ctx);
                        orswot.apply(op);
                    }
                    for i in 0..*size {
                        let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                        let op = orswot.rm(format!("value_{}", i), rm_ctx);
                        orswot.apply(op);
                    }
                })
            },
        );

        // Map Remove
        custom_benchmark(&mut group, &format!("Map Multiple Remove {}", size), || {
            Box::new(move || {
                let mut map: TestMap = Map::new();
                for i in 0..*size {
                    let add_ctx = map.read_ctx().derive_add_ctx(1);
                    let op = map.update(i as u32, add_ctx, |v, ctx| v.write(i * 10, ctx));
                    map.apply(op);
                }
                for i in 0..*size {
                    let rm_ctx = map.read_ctx().derive_rm_ctx();
                    let op = map.rm(i as u32, rm_ctx);
                    map.apply(op);
                }
            })
        });

        // List Remove
        custom_benchmark(
            &mut group,
            &format!("List Multiple Remove {}", size),
            || {
                Box::new(move || {
                    let mut list = List::new();
                    for i in 0..*size {
                        let op = list.insert_index(
                            i.try_into().unwrap(),
                            format!("value_{}", i),
                            "actor1",
                        );
                        list.apply(op);
                    }
                    for i in (0..*size).rev() {
                        let op = list.delete_index(i.try_into().unwrap(), "actor1");
                        list.apply(op.unwrap());
                    }
                })
            },
        );
    }

    group.finish();
}

fn concurrent_remove_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Remove Single Operations");
    const NUM_THREADS: usize = 4;

    // Initialize CRDTs wrapped in Arc<Mutex<>>
    let orswot = Arc::new(Mutex::new(Orswot::new()));
    let pncounter = Arc::new(Mutex::new(PNCounter::new()));
    let map: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));
    let list = Arc::new(Mutex::new(List::new()));

    // Pre-insert values for removal to work
    {
        let mut orswot = orswot.lock().unwrap();
        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
        let add_op = orswot.add("value", add_ctx);
        orswot.apply(add_op);
    }
    {
        let mut map = map.lock().unwrap();
        let actor = 1;
        let add_ctx = map.read_ctx().derive_add_ctx(actor);
        let op = map.update(black_box(1 as u32), add_ctx, |v, ctx| {
            v.write(black_box(42), ctx)
        });
        map.apply(op);
    }
    {
        let mut pncounter = pncounter.lock().unwrap();
        pncounter.inc(NUM_THREADS as u64);
    }
    {
        let mut list = list.lock().unwrap();
        for _ in 0..NUM_THREADS {
            let op = list.insert_index(black_box(0), black_box("element"), black_box("actor"));
            list.apply(op);
        }
    }

    // Custom benchmark for Concurrent ORSWOT Remove
    custom_benchmark(&mut group, "Concurrent Orswot Remove", || {
        let orswot = Arc::clone(&orswot);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|_| {
                    thread::spawn({
                        let value = orswot.clone();
                        move || {
                            let mut orswot = value.lock().unwrap();
                            let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                            let rm_op = orswot.rm("value", rm_ctx);
                            orswot.apply(rm_op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // // Custom benchmark for Concurrent PNCounter Decrement
    // custom_benchmark(&mut group, "Concurrent PNCounter Decrement", || {
    //     let pncounter = Arc::clone(&pncounter);
    //     Box::new(move || {
    //         let handles: Vec<_> = (0..NUM_THREADS)
    //             .map(|_| {
    //                 thread::spawn({
    //                     let value = pncounter.clone();
    //                     move || {
    //                         let mut pncounter = value.lock().unwrap();
    //                         pncounter.dec(black_box(1));
    //                     }
    //                 })
    //             })
    //             .collect();
    //         for handle in handles {
    //             handle.join().unwrap();
    //         }
    //     })
    // });

    // Custom benchmark for Concurrent Map Remove
    custom_benchmark(&mut group, "Concurrent Map Remove", || {
        let map = Arc::clone(&map);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|_| {
                    thread::spawn({
                        let value = map.clone();
                        move || {
                            let mut map = value.lock().unwrap();
                            let rm_ctx = map.read_ctx().derive_rm_ctx();
                            let op = map.rm(black_box(1 as u32), rm_ctx);
                            map.apply(op);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Custom benchmark for Concurrent List Remove
    custom_benchmark(&mut group, "Concurrent List Remove", || {
        let list = Arc::clone(&list);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|thread_id| {
                    thread::spawn({
                        let value = list.clone();
                        move || {
                            let mut list = value.lock().unwrap();
                            if let Some(del_op) =
                                list.delete_index(black_box(thread_id), black_box("actor"))
                            {
                                list.apply(del_op);
                            }
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    group.finish();
}

fn concurrent_remove_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Remove Multiple Operations");
    const NUM_THREADS: usize = 4;

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in sizes.iter() {
        let elements_per_thread = size / NUM_THREADS;

        // Concurrent ORSWOT Remove Multiple
        custom_benchmark(
            &mut group,
            &format!("Concurrent Orswot Remove Multiple {}", size),
            || {
                let orswot = Arc::new(Mutex::new(Orswot::new()));
                // Pre-insert values
                {
                    let mut orswot = orswot.lock().unwrap();
                    for i in 0..*size {
                        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                        let add_op = orswot.add(format!("value{}", i), add_ctx);
                        orswot.apply(add_op);
                    }
                }
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let orswot = Arc::clone(&orswot);
                            thread::spawn(move || {
                                let mut orswot = orswot.lock().unwrap();
                                for i in 0..elements_per_thread {
                                    let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                                    let rm_op = orswot.rm(
                                        format!("value{}", thread_id * elements_per_thread + i),
                                        rm_ctx,
                                    );
                                    orswot.apply(rm_op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Concurrent PNCounter Decrement Multiple
        custom_benchmark(
            &mut group,
            &format!("Concurrent PNCounter Decrement Multiple {}", size),
            || {
                let pncounter = Arc::new(Mutex::new(PNCounter::new()));
                // Pre-increment counter
                {
                    let mut pncounter = pncounter.lock().unwrap();
                    pncounter.inc(*size as u64);
                }
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            let pncounter = Arc::clone(&pncounter);
                            thread::spawn(move || {
                                let mut pncounter = pncounter.lock().unwrap();
                                pncounter.dec(elements_per_thread as u64);
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Concurrent Map Remove Multiple
        custom_benchmark(
            &mut group,
            &format!("Concurrent Map Remove Multiple {}", size),
            || {
                let map: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));
                // Pre-insert key-value pairs
                {
                    let mut map = map.lock().unwrap();
                    for i in 0..*size {
                        let add_ctx = map.read_ctx().derive_add_ctx(1);
                        let op = map.update(i as u32, add_ctx, |v, ctx| v.write(i as u32, ctx));
                        map.apply(op);
                    }
                }
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let map = Arc::clone(&map);
                            thread::spawn(move || {
                                let mut map = map.lock().unwrap();
                                for i in 0..elements_per_thread {
                                    let rm_ctx = map.read_ctx().derive_rm_ctx();
                                    let op = map
                                        .rm((thread_id * elements_per_thread + i) as u32, rm_ctx);
                                    map.apply(op);
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Concurrent List Remove Multiple
        custom_benchmark(
            &mut group,
            &format!("Concurrent List Remove Multiple {}", size),
            || {
                let list = Arc::new(Mutex::new(List::new()));
                // Pre-insert elements
                {
                    let mut list = list.lock().unwrap();
                    for i in 0..*size {
                        let op = list.insert_index(i, format!("element{}", i), "actor");
                        list.apply(op);
                    }
                }
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let list = Arc::clone(&list);
                            thread::spawn(move || {
                                let mut list = list.lock().unwrap();
                                for _ in 0..elements_per_thread {
                                    if let Some(del_op) =
                                        list.delete_index(thread_id * elements_per_thread, "actor")
                                    {
                                        list.apply(del_op);
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

fn merge_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merge Single Operations");

    custom_benchmark(&mut group, "GSet Single Merge", || {
        Box::new(|| {
            let mut gset1 = GSet::new();
            let mut gset2 = GSet::new();
            gset1.insert(1);
            gset2.insert(2);
            black_box(gset1.merge(gset2));
        })
    });

    custom_benchmark(&mut group, "Orswot Single Merge", || {
        Box::new(|| {
            let mut orswot1 = Orswot::new();
            let mut orswot2 = Orswot::new();
            let add_ctx1 = orswot1.read_ctx().derive_add_ctx(1);
            let add_ctx2 = orswot2.read_ctx().derive_add_ctx(2);
            let op1 = orswot1.add("value1", add_ctx1);
            let op2 = orswot2.add("value2", add_ctx2);
            orswot1.apply(op1);
            orswot2.apply(op2);
            black_box(orswot1.merge(orswot2));
        })
    });

    custom_benchmark(&mut group, "GCounter Single Merge", || {
        Box::new(|| {
            let mut gcounter1 = GCounter::new();
            let mut gcounter2 = GCounter::new();
            gcounter1.inc(5);
            gcounter2.inc(3);
            black_box(gcounter1.merge(gcounter2));
        })
    });

    custom_benchmark(&mut group, "PNCounter Single Merge", || {
        Box::new(|| {
            let mut pncounter1 = PNCounter::new();
            let mut pncounter2 = PNCounter::new();
            pncounter1.inc(5);
            pncounter2.dec(3);
            black_box(pncounter1.merge(pncounter2));
        })
    });

    custom_benchmark(&mut group, "LWWReg Single Merge", || {
        Box::new(|| {
            let mut lww_reg1 = LWWReg { val: 10, marker: 1 };
            let lww_reg2 = LWWReg { val: 20, marker: 2 };
            black_box(lww_reg1.merge(lww_reg2));
        })
    });

    custom_benchmark(&mut group, "MVReg Single Merge", || {
        Box::new(|| {
            let mut mvreg1 = MVReg::new();
            let mut mvreg2 = MVReg::new();
            let add_ctx1 = mvreg1.read_ctx().derive_add_ctx(1);
            let add_ctx2 = mvreg2.read_ctx().derive_add_ctx(2);
            let op1 = mvreg1.write("value1", add_ctx1);
            let op2 = mvreg2.write("value2", add_ctx2);
            mvreg1.apply(op1);
            mvreg2.apply(op2);
            black_box(mvreg1.merge(mvreg2));
        })
    });

    custom_benchmark(&mut group, "Map Single Merge", || {
        Box::new(|| {
            let mut map1: TestMap = Map::new();
            let mut map2: TestMap = Map::new();
            let add_ctx1 = map1.read_ctx().derive_add_ctx(1);
            let add_ctx2 = map2.read_ctx().derive_add_ctx(2);
            let op1 = map1.update(1 as u32, add_ctx1, |v, ctx| v.write(10, ctx));
            let op2 = map2.update(2 as u32, add_ctx2, |v, ctx| v.write(20, ctx));
            map1.apply(op1);
            map2.apply(op2);
            black_box(map1.merge(map2));
        })
    });

    custom_benchmark(&mut group, "GList Single Merge", || {
        Box::new(|| {
            let mut glist1 = GList::new();
            let mut glist2 = GList::new();
            let op1 = glist1.insert_after(None, "value1");
            let op2 = glist2.insert_after(None, "value2");
            glist1.apply(op1);
            glist2.apply(op2);
            black_box(glist1.merge(glist2));
        })
    });

    group.finish();
}

fn merge_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merge Multiple Operations");

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        custom_benchmark(&mut group, &format!("GSet Multiple Merge {}", size), || {
            Box::new(move || {
                let mut gset1 = GSet::new();
                let mut gset2 = GSet::new();
                for i in 0..*size {
                    gset1.insert(i * 2);
                    gset2.insert(i * 2 + 1);
                }
                black_box(gset1.merge(gset2));
            })
        });

        custom_benchmark(
            &mut group,
            &format!("Orswot Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mut orswot1 = Orswot::new();
                    let mut orswot2 = Orswot::new();
                    for i in 0..*size {
                        let add_ctx1 = orswot1.read_ctx().derive_add_ctx(1);
                        let add_ctx2 = orswot2.read_ctx().derive_add_ctx(2);
                        let op1 = orswot1.add(format!("value1_{}", i), add_ctx1);
                        let op2 = orswot2.add(format!("value2_{}", i), add_ctx2);
                        orswot1.apply(op1);
                        orswot2.apply(op2);
                    }
                    black_box(orswot1.merge(orswot2));
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GCounter Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mut gcounter1 = GCounter::new();
                    let mut gcounter2 = GCounter::new();
                    for i in 0..*size {
                        gcounter1.inc(i);
                        gcounter2.inc(i + 1);
                    }
                    black_box(gcounter1.merge(gcounter2));
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("PNCounter Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mut pncounter1 = PNCounter::new();
                    let mut pncounter2 = PNCounter::new();
                    for i in 0..*size {
                        pncounter1.inc(i);
                        pncounter2.dec(i);
                    }
                    black_box(pncounter1.merge(pncounter2));
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("LWWReg Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mut lww_reg1 = LWWReg { val: 0, marker: 0 };
                    let mut lww_reg2 = LWWReg { val: 0, marker: 0 };
                    for i in 0..*size {
                        lww_reg1.update(i as u64, i * 10);
                        lww_reg2.update(i as u64 + 1, i * 20);
                    }
                    black_box(lww_reg1.merge(lww_reg2));
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("MVReg Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mut mvreg1 = MVReg::new();
                    let mut mvreg2 = MVReg::new();
                    for i in 0..*size {
                        let add_ctx1 = mvreg1.read_ctx().derive_add_ctx(1);
                        let add_ctx2 = mvreg2.read_ctx().derive_add_ctx(2);
                        let op1 = mvreg1.write(format!("value1_{}", i), add_ctx1);
                        let op2 = mvreg2.write(format!("value2_{}", i), add_ctx2);
                        mvreg1.apply(op1);
                        mvreg2.apply(op2);
                    }
                    black_box(mvreg1.merge(mvreg2));
                })
            },
        );

        custom_benchmark(&mut group, &format!("Map Multiple Merge {}", size), || {
            Box::new(move || {
                let mut map1: TestMap = Map::new();
                let mut map2: TestMap = Map::new();
                for i in 0..*size {
                    let add_ctx1 = map1.read_ctx().derive_add_ctx(1);
                    let add_ctx2 = map2.read_ctx().derive_add_ctx(2);
                    let op1 = map1.update(i as u32, add_ctx1, |v, ctx| v.write(i * 10, ctx));
                    let op2 = map2.update((i + 1) as u32, add_ctx2, |v, ctx| v.write(i * 20, ctx));
                    map1.apply(op1);
                    map2.apply(op2);
                }
                black_box(map1.merge(map2));
            })
        });

        // custom_benchmark(&mut group, &format!("List Multiple Merge {}", size), || {
        //     Box::new(move || {
        //         let mut list1 = List::new();
        //         let mut list2 = List::new();
        //         for i in 0..*size {
        //             let op1 = list1.insert_index(i.try_into().unwrap(), i, "actor1");
        //             let op2 = list2.insert_index(i.try_into().unwrap(), i + 1, "actor2");
        //             list1.apply(op1);
        //             list2.apply(op2);
        //         }
        //         black_box(list1.merge(list2));
        //     })
        // });

        custom_benchmark(
            &mut group,
            &format!("GList Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mut glist1 = GList::new();
                    let mut glist2 = GList::new();
                    for i in 0..*size {
                        let op1 = glist1.insert_after(None, format!("value1_{}", i));
                        let op2 = glist2.insert_after(None, format!("value2_{}", i));
                        glist1.apply(op1);
                        glist2.apply(op2);
                    }
                    black_box(glist1.merge(glist2));
                })
            },
        );
    }

    group.finish();
}

fn concurrent_merge_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Merge Single Operations");
    const NUM_THREADS: usize = 4;

    let gset1 = Arc::new(Mutex::new(GSet::<String>::new()));
    let gset2 = Arc::new(Mutex::new(GSet::<String>::new()));

    let orswot1 = Arc::new(Mutex::new(Orswot::<String, u64>::new()));
    let orswot2 = Arc::new(Mutex::new(Orswot::<String, u64>::new()));

    let gcounter1 = Arc::new(Mutex::new(GCounter::<usize>::new()));
    let gcounter2 = Arc::new(Mutex::new(GCounter::<usize>::new()));

    let pncounter1 = Arc::new(Mutex::new(PNCounter::<usize>::new()));
    let pncounter2 = Arc::new(Mutex::new(PNCounter::<usize>::new()));

    let lww_reg1 = Arc::new(Mutex::new(LWWReg { val: 42, marker: 0 }));
    let lww_reg2 = Arc::new(Mutex::new(LWWReg { val: 84, marker: 1 }));

    let mv_reg1 = Arc::new(Mutex::new(MVReg::<String, u64>::new()));
    let mv_reg2 = Arc::new(Mutex::new(MVReg::<String, u64>::new()));

    // type TestMap = Map<u32, LWWReg<u32, Dot<u64>>, u64>;
    let map1: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));
    let map2: Arc<Mutex<TestMap>> = Arc::new(Mutex::new(Map::new()));

    let glist1 = Arc::new(Mutex::new(GList::<String>::new()));
    let glist2 = Arc::new(Mutex::new(GList::<String>::new()));

    // Benchmark for Concurrent GSet Merge
    custom_benchmark(&mut group, "Concurrent GSet Merge", || {
        let gset1 = Arc::clone(&gset1);
        let gset2 = Arc::clone(&gset2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value1 = gset1.clone();
                        let value2 = gset2.clone();
                        move || {
                            let mut set1 = value1.lock().unwrap();
                            let mut set2 = value2.lock().unwrap();
                            set1.insert(black_box(format!("element1_{}", i)));
                            set2.insert(black_box(format!("element2_{}", i)));
                            black_box(set1.merge(set2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent Orswot Merge
    custom_benchmark(&mut group, "Concurrent Orswot Merge", || {
        let orswot1 = Arc::clone(&orswot1);
        let orswot2 = Arc::clone(&orswot2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value1 = orswot1.clone();
                        let value2 = orswot2.clone();
                        move || {
                            let mut orswot1 = value1.lock().unwrap();
                            let mut orswot2 = value2.lock().unwrap();
                            let add_ctx1 = orswot1.read_ctx().derive_add_ctx(i as u64);
                            let add_ctx2 =
                                orswot2.read_ctx().derive_add_ctx((i + NUM_THREADS) as u64);
                            let op1 = orswot1.add(black_box(format!("value1_{}", i)), add_ctx1);
                            let op2 = orswot2.add(black_box(format!("value2_{}", i)), add_ctx2);
                            orswot1.apply(op1);
                            orswot2.apply(op2);
                            black_box(orswot1.merge(orswot2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent GCounter Merge
    custom_benchmark(&mut group, "Concurrent GCounter Merge", || {
        let gcounter1 = Arc::clone(&gcounter1);
        let gcounter2 = Arc::clone(&gcounter2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|_| {
                    thread::spawn({
                        let value1 = gcounter1.clone();
                        let value2 = gcounter2.clone();
                        move || {
                            let mut counter1 = value1.lock().unwrap();
                            let mut counter2 = value2.lock().unwrap();
                            counter1.inc(black_box(1));
                            counter2.inc(black_box(2));
                            black_box(counter1.merge(counter2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent PNCounter Merge
    custom_benchmark(&mut group, "Concurrent PNCounter Merge", || {
        let pncounter1 = Arc::clone(&pncounter1);
        let pncounter2 = Arc::clone(&pncounter2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|_| {
                    thread::spawn({
                        let value1 = pncounter1.clone();
                        let value2 = pncounter2.clone();
                        move || {
                            let mut counter1 = value1.lock().unwrap();
                            let mut counter2 = value2.lock().unwrap();
                            counter1.inc(black_box(1));
                            counter2.dec(black_box(1));
                            black_box(counter1.merge(counter2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent LWWReg Merge
    custom_benchmark(&mut group, "Concurrent LWWReg Merge", || {
        let lww_reg1 = Arc::clone(&lww_reg1);
        let lww_reg2 = Arc::clone(&lww_reg2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value1 = lww_reg1.clone();
                        let value2 = lww_reg2.clone();
                        move || {
                            let mut reg1 = value1.lock().unwrap();
                            let mut reg2 = value2.lock().unwrap();
                            reg1.update(black_box(100 + i as u64), black_box(i as u64));
                            reg2.update(
                                black_box(200 + i as u64),
                                black_box((i + NUM_THREADS) as u64),
                            );
                            black_box(reg1.merge(reg2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent MVReg Merge
    custom_benchmark(&mut group, "Concurrent MVReg Merge", || {
        let mv_reg1 = Arc::clone(&mv_reg1);
        let mv_reg2 = Arc::clone(&mv_reg2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value1 = mv_reg1.clone();
                        let value2 = mv_reg2.clone();
                        move || {
                            let mut reg1 = value1.lock().unwrap();
                            let mut reg2 = value2.lock().unwrap();
                            let add_ctx1 = reg1.read_ctx().derive_add_ctx(i as u64);
                            let add_ctx2 = reg2.read_ctx().derive_add_ctx((i + NUM_THREADS) as u64);
                            let op1 = reg1.write(black_box(format!("value1_{}", i)), add_ctx1);
                            let op2 = reg2.write(black_box(format!("value2_{}", i)), add_ctx2);
                            reg1.apply(op1);
                            reg2.apply(op2);
                            black_box(reg1.merge(reg2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent Map Merge
    custom_benchmark(&mut group, "Concurrent Map Merge", || {
        let map1 = Arc::clone(&map1);
        let map2 = Arc::clone(&map2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value1 = map1.clone();
                        let value2 = map2.clone();
                        move || {
                            let mut map1 = value1.lock().unwrap();
                            let mut map2 = value2.lock().unwrap();
                            let add_ctx1 = map1
                                .read_ctx()
                                .derive_add_ctx((i as u64).try_into().unwrap());
                            let add_ctx2 = map2
                                .read_ctx()
                                .derive_add_ctx(((i + NUM_THREADS) as u64).try_into().unwrap());
                            let op1 = map1
                                .update(i as u32, add_ctx1, |v, ctx| v.write((i * 10) as u32, ctx));
                            let op2 = map2.update((i + NUM_THREADS) as u32, add_ctx2, |v, ctx| {
                                v.write((i * 20) as u32, ctx)
                            });
                            map1.apply(op1);
                            map2.apply(op2);
                            black_box(map1.merge(map2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Benchmark for Concurrent GList Merge
    custom_benchmark(&mut group, "Concurrent GList Merge", || {
        let glist1 = Arc::clone(&glist1);
        let glist2 = Arc::clone(&glist2);
        Box::new(move || {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|i| {
                    thread::spawn({
                        let value1 = glist1.clone();
                        let value2 = glist2.clone();
                        move || {
                            let mut list1 = value1.lock().unwrap();
                            let mut list2 = value2.lock().unwrap();
                            let op1 =
                                list1.insert_after(None, black_box(format!("element1_{}", i)));
                            let op2 =
                                list2.insert_after(None, black_box(format!("element2_{}", i)));
                            list1.apply(op1);
                            list2.apply(op2);
                            black_box(list1.merge(list2.clone()));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    group.finish();
}

fn concurrent_merge_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Merge Multiple Operations");

    const NUM_THREADS: usize = 4;
    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        let instances_per_thread = size / NUM_THREADS;

        // Helper function to create Vec of Arc<Mutex<T>>
        fn create_instances<T: Clone + Default>(count: usize) -> Vec<Arc<Mutex<T>>> {
            (0..count)
                .map(|_| Arc::new(Mutex::new(T::default())))
                .collect()
        }

        custom_benchmark(
            &mut group,
            &format!("Concurrent GSet Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let gsets: Vec<Arc<Mutex<GSet<String>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let gsets = gsets.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut gset =
                                        gsets[thread_id * instances_per_thread + i].lock().unwrap();
                                    gset.insert(black_box(format!("element{}_{}", i, thread_id)));
                                    if i > 0 {
                                        let prev_gset = gsets
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(gset.merge(prev_gset.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent Orswot Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let orswots: Vec<Arc<Mutex<Orswot<String, u64>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let orswots = orswots.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut orswot = orswots[thread_id * instances_per_thread + i]
                                        .lock()
                                        .unwrap();
                                    let add_ctx =
                                        orswot.read_ctx().derive_add_ctx(thread_id as u64);
                                    let op = orswot.add(
                                        black_box(format!("value{}_{}", i, thread_id)),
                                        add_ctx,
                                    );
                                    orswot.apply(op);
                                    if i > 0 {
                                        let prev_orswot = orswots
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(orswot.merge(prev_orswot.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent GCounter Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let gcounters: Vec<Arc<Mutex<GCounter<usize>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let gcounters = gcounters.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut gcounter = gcounters
                                        [thread_id * instances_per_thread + i]
                                        .lock()
                                        .unwrap();
                                    gcounter.inc(black_box(thread_id + 1));
                                    if i > 0 {
                                        let prev_gcounter = gcounters
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(gcounter.merge(prev_gcounter.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent PNCounter Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let pncounters: Vec<Arc<Mutex<PNCounter<usize>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let pncounters = pncounters.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut pncounter = pncounters
                                        [thread_id * instances_per_thread + i]
                                        .lock()
                                        .unwrap();
                                    if thread_id % 2 == 0 {
                                        pncounter.inc(black_box(1));
                                    } else {
                                        pncounter.dec(black_box(1));
                                    }
                                    if i > 0 {
                                        let prev_pncounter = pncounters
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(pncounter.merge(prev_pncounter.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent LWWReg Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let lww_regs: Vec<Arc<Mutex<LWWReg<u64, u64>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let lww_regs = lww_regs.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut lww_reg = lww_regs
                                        [thread_id * instances_per_thread + i]
                                        .lock()
                                        .unwrap();
                                    lww_reg.update(
                                        black_box(thread_id as u64),
                                        black_box((100 + i) as u64),
                                    );
                                    if i > 0 {
                                        let prev_lww_reg = lww_regs
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(lww_reg.merge(prev_lww_reg.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent MVReg Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let mv_regs: Vec<Arc<Mutex<MVReg<String, u64>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let mv_regs = mv_regs.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut mv_reg = mv_regs[thread_id * instances_per_thread + i]
                                        .lock()
                                        .unwrap();
                                    let add_ctx =
                                        mv_reg.read_ctx().derive_add_ctx(thread_id as u64);
                                    let op = mv_reg.write(
                                        black_box(format!("value{}_{}", i, thread_id)),
                                        add_ctx,
                                    );
                                    mv_reg.apply(op);
                                    if i > 0 {
                                        let prev_mv_reg = mv_regs
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(mv_reg.merge(prev_mv_reg.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent Map Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let maps: Vec<Arc<Mutex<TestMap>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let maps = maps.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut map =
                                        maps[thread_id * instances_per_thread + i].lock().unwrap();
                                    let add_ctx = map
                                        .read_ctx()
                                        .derive_add_ctx((thread_id as u64).try_into().unwrap());
                                    let op = map.update(thread_id as u32, add_ctx, |v, ctx| {
                                        v.write((i * 10).try_into().unwrap_or_default(), ctx)
                                    });
                                    map.apply(op);
                                    if i > 0 {
                                        let prev_map = maps
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(map.merge(prev_map.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Concurrent GList Multiple Merge {}", size),
            || {
                Box::new(move || {
                    let glists: Vec<Arc<Mutex<GList<String>>>> = create_instances(*size);
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|thread_id| {
                            let glists = glists.clone();
                            thread::spawn(move || {
                                for i in 0..instances_per_thread {
                                    let mut glist = glists[thread_id * instances_per_thread + i]
                                        .lock()
                                        .unwrap();
                                    let op = glist.insert_after(
                                        None,
                                        black_box(format!("element{}_{}", i, thread_id)),
                                    );
                                    glist.apply(op);
                                    if i > 0 {
                                        let prev_glist = glists
                                            [thread_id * instances_per_thread + i - 1]
                                            .lock()
                                            .unwrap();
                                        black_box(glist.merge(prev_glist.clone()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

fn read_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Read Operations");

    // Initialize CRDTs with some data (same as before)
    let mut gset = GSet::new();
    gset.insert("element");

    let orswot = Orswot::new();
    let add_ctx = orswot.read_ctx().derive_add_ctx(1);
    orswot.add("value", add_ctx);

    let gcounter = GCounter::new();
    gcounter.inc(1);

    let pncounter = PNCounter::new();
    pncounter.inc(1);

    let lww_reg = LWWReg { val: 42, marker: 0 };

    let mv_reg = MVReg::new();
    let read_ctx = mv_reg.read_ctx();
    let add_ctx = read_ctx.derive_add_ctx(1);
    mv_reg.write("value", add_ctx);

    let map: TestMap = Map::new();
    let add_ctx = map.read_ctx().derive_add_ctx(1);
    map.update(1 as u32, add_ctx, |v, ctx| v.write(42, ctx));

    let list = List::new();
    list.insert_index(0, "element", "actor");

    let glist = GList::new();
    glist.insert_after(None, "element");

    // Benchmark for GSet Read
    custom_benchmark(&mut group, "GSet Read", || {
        Box::new({
            let value = gset.clone();
            move || {
                black_box(value.contains(&"element"));
            }
        })
    });

    // Benchmark for ORSWOT Read
    custom_benchmark(&mut group, "Orswot Read", || {
        Box::new({
            let value = orswot.clone();
            move || {
                black_box(value.contains(&"value"));
            }
        })
    });

    // Benchmark for GCounter Read
    custom_benchmark(&mut group, "GCounter Read", || {
        Box::new({
            let value = gcounter.clone();
            move || {
                black_box(value.read());
            }
        })
    });

    // Benchmark for PNCounter Read
    custom_benchmark(&mut group, "PNCounter Read", || {
        Box::new({
            let value = pncounter.clone();
            move || {
                black_box(value.read());
            }
        })
    });

    // Benchmark for LWWReg Read
    custom_benchmark(&mut group, "LWWReg Read", || {
        Box::new(move || {
            black_box(lww_reg.val);
        })
    });

    // Benchmark for MVReg Read
    custom_benchmark(&mut group, "MVReg Read", || {
        Box::new({
            let value = mv_reg.clone();
            move || {
                black_box(value.read());
            }
        })
    });

    // Benchmark for Map Read
    custom_benchmark(&mut group, "Map Read", || {
        Box::new({
            let value = map.clone();
            move || {
                black_box(value.get(&1));
            }
        })
    });

    // Benchmark for List Read
    custom_benchmark(&mut group, "List Read", || {
        Box::new({
            let value = list.clone();
            move || {
                black_box(value.position(0));
            }
        })
    });

    // Benchmark for GList Read
    custom_benchmark(&mut group, "GList Read", || {
        Box::new({
            let value = glist.clone();
            move || {
                black_box(value.get(0));
            }
        })
    });

    group.finish();
}

fn read_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Read Multiple Operations");

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        // GSet Read
        custom_benchmark(&mut group, &format!("GSet Multiple Read {}", size), || {
            Box::new(move || {
                let mut gset = GSet::new();
                for i in 0..*size {
                    gset.insert(i);
                }
                for i in 0..*size {
                    black_box(gset.contains(&i));
                }
            })
        });

        // Orswot Read
        custom_benchmark(
            &mut group,
            &format!("Orswot Multiple Read {}", size),
            || {
                Box::new(move || {
                    let mut orswot = Orswot::new();
                    for i in 0..*size {
                        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                        let op = orswot.add(format!("value_{}", i), add_ctx);
                        orswot.apply(op);
                    }
                    for i in 0..*size {
                        black_box(orswot.contains(&format!("value_{}", i)));
                    }
                })
            },
        );

        // GCounter Read
        custom_benchmark(
            &mut group,
            &format!("GCounter Multiple Read {}", size),
            || {
                Box::new(move || {
                    let mut gcounter = GCounter::new();
                    for i in 0..*size {
                        gcounter.inc(i);
                    }
                    black_box(gcounter.read());
                })
            },
        );

        // PNCounter Read
        custom_benchmark(
            &mut group,
            &format!("PNCounter Multiple Read {}", size),
            || {
                Box::new(move || {
                    let mut pncounter = PNCounter::new();
                    for i in 0..*size {
                        pncounter.inc(i);
                        pncounter.dec(i / 2);
                    }
                    black_box(pncounter.read());
                })
            },
        );

        // LWWReg Read
        custom_benchmark(
            &mut group,
            &format!("LWWReg Multiple Read {}", size),
            || {
                Box::new(move || {
                    let mut lww_reg = LWWReg { val: 0, marker: 0 };
                    for i in 0..*size {
                        lww_reg.update(i as u64, i * 10);
                    }
                    black_box(lww_reg.val);
                })
            },
        );

        // MVReg Read
        custom_benchmark(&mut group, &format!("MVReg Multiple Read {}", size), || {
            Box::new(move || {
                let mut mvreg = MVReg::new();
                for i in 0..*size {
                    let add_ctx = mvreg.read_ctx().derive_add_ctx(1);
                    let op = mvreg.write(format!("value_{}", i), add_ctx);
                    mvreg.apply(op);
                }
                black_box(mvreg.read());
            })
        });

        // Map Read
        custom_benchmark(&mut group, &format!("Map Multiple Read {}", size), || {
            Box::new(move || {
                let mut map: TestMap = Map::new();
                for i in 0..*size {
                    let add_ctx = map.read_ctx().derive_add_ctx(1);
                    let op = map.update(i as u32, add_ctx, |v, ctx| v.write(i * 10, ctx));
                    map.apply(op);
                }
                for i in 0..*size {
                    black_box(map.get(&(i as u32)));
                }
            })
        });

        // GList Read
        custom_benchmark(&mut group, &format!("GList Multiple Read {}", size), || {
            Box::new(move || {
                let mut glist = GList::new();
                for i in 0..*size {
                    let op = glist.insert_after(None, format!("value_{}", i));
                    glist.apply(op);
                }
                for i in 0..*size {
                    black_box(glist.get(i.try_into().unwrap()));
                }
            })
        });

        // List Read
        custom_benchmark(&mut group, &format!("List Multiple Read {}", size), || {
            Box::new(move || {
                let mut list = List::new();
                for i in 0..*size {
                    let op =
                        list.insert_index(i.try_into().unwrap(), format!("value_{}", i), "actor1");
                    list.apply(op);
                }
                for i in 0..*size {
                    black_box(list.position(i.try_into().unwrap()));
                }
            })
        });
    }

    group.finish();
}

fn concurrent_read_single_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Read Single Operations");
    const NUM_THREADS: usize = 4;

    // Initialize CRDTs with some data
    let gset = Arc::new(Mutex::new({
        let mut set = GSet::new();
        set.insert("element");
        set
    }));
    let orswot = Arc::new(Mutex::new({
        let set = Orswot::new();
        set.add("value", set.read_ctx().derive_add_ctx(1));
        set
    }));
    let gcounter = Arc::new(Mutex::new({
        let counter = GCounter::new();
        counter.inc(1);
        counter
    }));
    let pncounter = Arc::new(Mutex::new({
        let counter = PNCounter::new();
        counter.inc(1);
        counter
    }));
    let lww_reg = Arc::new(Mutex::new(LWWReg { val: 42, marker: 0 }));
    let mv_reg = Arc::new(Mutex::new({
        let reg = MVReg::new();
        reg.write("value", reg.read_ctx().derive_add_ctx(1));
        reg
    }));
    let list = Arc::new(Mutex::new({
        let l = List::new();
        l.insert_index(0, "element", "actor");
        l
    }));
    let glist = Arc::new(Mutex::new({
        let l = GList::new();
        l.insert_after(None, "element");
        l
    }));
    let map = Arc::new(Mutex::new({
        let mut map: TestMap = Map::new();

        let add_ctx = map
            .read_ctx()
            .derive_add_ctx((1 as u64).try_into().unwrap());
        let op = map.update(1 as u32, add_ctx, |v, ctx| v.write(1 as u32, ctx));
        map.apply(op);

        map
    }));

    // Benchmark for Concurrent GSet Read
    custom_benchmark(&mut group, "Concurrent GSet Read", || {
        Box::new({
            let value = gset.clone();
            move || {
                let gset = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = gset.clone();
                            move || {
                                let gset = value.lock().unwrap();
                                black_box(gset.contains(&"element"));
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent ORSWOT Read
    custom_benchmark(&mut group, "Concurrent Orswot Read", || {
        Box::new({
            let value = orswot.clone();
            move || {
                let orswot = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = orswot.clone();
                            move || {
                                let orswot = value.lock().unwrap();
                                black_box(orswot.contains(&"value"));
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent GCounter Read
    custom_benchmark(&mut group, "Concurrent GCounter Read", || {
        Box::new({
            let value = gcounter.clone();
            move || {
                let gcounter = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = gcounter.clone();
                            move || {
                                let gcounter = value.lock().unwrap();
                                black_box(gcounter.read());
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent PNCounter Read
    custom_benchmark(&mut group, "Concurrent PNCounter Read", || {
        Box::new({
            let value = pncounter.clone();
            move || {
                let pncounter = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = pncounter.clone();
                            move || {
                                let pncounter = value.lock().unwrap();
                                black_box(pncounter.read());
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent LWWReg Read
    custom_benchmark(&mut group, "Concurrent LWWReg Read", || {
        Box::new({
            let value = lww_reg.clone();
            move || {
                let lww_reg = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = lww_reg.clone();
                            move || {
                                let lww_reg = value.lock().unwrap();
                                black_box(lww_reg.val);
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent MVReg Read
    custom_benchmark(&mut group, "Concurrent MVReg Read", || {
        Box::new({
            let value = mv_reg.clone();
            move || {
                let mv_reg = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = mv_reg.clone();
                            move || {
                                let mv_reg = value.lock().unwrap();
                                black_box(mv_reg.read());
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent List Read
    custom_benchmark(&mut group, "Concurrent List Read", || {
        Box::new({
            let value = list.clone();
            move || {
                let list = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = list.clone();
                            move || {
                                let list = value.lock().unwrap();
                                black_box(list.position(0));
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    // Benchmark for Concurrent GList Read
    custom_benchmark(&mut group, "Concurrent GList Read", || {
        Box::new({
            let value = glist.clone();
            move || {
                let glist = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = glist.clone();
                            move || {
                                let glist = value.lock().unwrap();
                                black_box(glist.get(0));
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    custom_benchmark(&mut group, "Concurrent Map Read", || {
        Box::new({
            let value = map.clone();
            move || {
                let map = Arc::clone(&value);
                let handles: Vec<_> = (0..NUM_THREADS)
                    .map(|_| {
                        thread::spawn({
                            let value = map.clone();
                            move || {
                                let map = value.lock().unwrap();
                                black_box(map.get(&1));
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            }
        })
    });

    group.finish();
}

// fn concurrent_read_multiple_benchmark(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Concurrent Read Multiple Operations");
//     group.measurement_time(Duration::from_secs(10));
//     const NUM_THREADS: usize = 4;
//     const NUM_READS: usize = 100;

//     // Initialize CRDTs with multiple elements
//     let gset = Arc::new(Mutex::new({
//         let mut set = GSet::new();
//         for i in 0..NUM_READS {
//             set.insert(i);
//         }
//         set
//     }));
//     let orswot = Arc::new(Mutex::new({
//         let mut set = Orswot::new();
//         for i in 0..NUM_READS {
//             let add_ctx = set.read_ctx().derive_add_ctx(i as u64);
//             let add_op = set.add(i, add_ctx);
//             set.apply(add_op);
//         }
//         set
//     }));
//     let gcounter = Arc::new(Mutex::new({
//         let mut counter = GCounter::new();
//         for _ in 0..NUM_READS {
//             counter.inc(1);
//         }
//         counter
//     }));
//     let pncounter = Arc::new(Mutex::new({
//         let mut counter = PNCounter::new();
//         for i in 0..NUM_READS {
//             if i % 2 == 0 {
//                 counter.inc(1);
//             } else {
//                 counter.dec(1);
//             }
//         }
//         counter
//     }));
//     let lwwreg = Arc::new(Mutex::new({
//         let mut reg = LWWReg { val: 42, marker: 0 };
//         for i in 0..NUM_READS {
//             reg.update(i, i);
//         }
//         reg
//     }));
//     let mvreg = Arc::new(Mutex::new({
//         let mut reg = MVReg::new();
//         for i in 0..NUM_READS {
//             let read_ctx = reg.read_ctx();
//             let add_ctx = read_ctx.derive_add_ctx(i as u64);
//             let op = reg.write(i, add_ctx);
//             reg.apply(op);
//         }
//         reg
//     }));
//     let map = Arc::new(Mutex::new({
//         let mut map: TestMap = Map::new();
//         for i in 0..NUM_READS {
//             let add_ctx = map
//                 .read_ctx()
//                 .derive_add_ctx((i as u64).try_into().unwrap());
//             let op = map.update(i as u32, add_ctx, |v, ctx| v.write(i as u32, ctx));
//             map.apply(op);
//         }
//         map
//     }));
//     let list = Arc::new(Mutex::new({
//         let mut list = List::new();
//         for i in 0..NUM_READS {
//             let op = list.insert_index(i, i, i.to_string());
//             list.apply(op);
//         }
//         list
//     }));
//     let glist = Arc::new(Mutex::new({
//         let mut list = GList::new();
//         for i in 0..NUM_READS {
//             let op = list.insert_after(None, i);
//             list.apply(op);
//         }
//         list
//     }));

//     // Custom benchmark for Concurrent GSet Multiple Reads
//     custom_benchmark(&mut group, "Concurrent GSet Multiple Reads", || {
//         let gset = Arc::clone(&gset);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = gset.clone();
//                         move || {
//                             let gset = value.lock().unwrap();
//                             for i in 0..NUM_READS {
//                                 black_box(gset.contains(&i));
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent ORSWOT Multiple Reads
//     custom_benchmark(&mut group, "Concurrent Orswot Multiple Reads", || {
//         let orswot = Arc::clone(&orswot);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = orswot.clone();
//                         move || {
//                             let orswot = value.lock().unwrap();
//                             for i in 0..NUM_READS {
//                                 black_box(orswot.contains(&i));
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent GCounter Multiple Reads
//     custom_benchmark(&mut group, "Concurrent GCounter Multiple Reads", || {
//         let gcounter = Arc::clone(&gcounter);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = gcounter.clone();
//                         move || {
//                             let gcounter = value.lock().unwrap();
//                             for _ in 0..NUM_READS {
//                                 black_box(gcounter.read());
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent PNCounter Multiple Reads
//     custom_benchmark(&mut group, "Concurrent PNCounter Multiple Reads", || {
//         let pncounter = Arc::clone(&pncounter);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = pncounter.clone();
//                         move || {
//                             let pncounter = value.lock().unwrap();
//                             for _ in 0..NUM_READS {
//                                 black_box(pncounter.read());
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent LWWReg Multiple Reads
//     custom_benchmark(&mut group, "Concurrent LWWReg Multiple Reads", || {
//         let lwwreg = Arc::clone(&lwwreg);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = lwwreg.clone();
//                         move || {
//                             let lwwreg = value.lock().unwrap();
//                             for _ in 0..NUM_READS {
//                                 black_box(lwwreg.val);
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent MVReg Multiple Reads
//     custom_benchmark(&mut group, "Concurrent MVReg Multiple Reads", || {
//         let mvreg = Arc::clone(&mvreg);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = mvreg.clone();
//                         move || {
//                             let mvreg = value.lock().unwrap();
//                             for _ in 0..NUM_READS {
//                                 black_box(mvreg.read());
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent Map Multiple Reads
//     custom_benchmark(&mut group, "Concurrent Map Multiple Reads", || {
//         let map = Arc::clone(&map);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = map.clone();
//                         move || {
//                             let map = value.lock().unwrap();
//                             for i in 0..NUM_READS {
//                                 black_box(map.get(&i.try_into().unwrap()));
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent List Multiple Reads
//     custom_benchmark(&mut group, "Concurrent List Multiple Reads", || {
//         let list = Arc::clone(&list);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = list.clone();
//                         move || {
//                             let list = value.lock().unwrap();
//                             for i in 0..NUM_READS {
//                                 black_box(list.position(i));
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     // Custom benchmark for Concurrent GList Multiple Reads
//     custom_benchmark(&mut group, "Concurrent GList Multiple Reads", || {
//         let glist = Arc::clone(&glist);
//         Box::new(move || {
//             let handles: Vec<_> = (0..NUM_THREADS)
//                 .map(|_| {
//                     thread::spawn({
//                         let value = glist.clone();
//                         move || {
//                             let glist = value.lock().unwrap();
//                             for i in 0..NUM_READS {
//                                 black_box(glist.get(i));
//                             }
//                         }
//                     })
//                 })
//                 .collect();
//             for handle in handles {
//                 handle.join().unwrap();
//             }
//         })
//     });

//     group.finish();
// }

fn concurrent_read_multiple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Read Multiple Operations");
    group.measurement_time(Duration::from_secs(10));
    const NUM_THREADS: usize = 4;

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        // Initialize CRDTs with multiple elements
        let gset = Arc::new(Mutex::new({
            let mut set = GSet::new();
            for i in 0..*size {
                set.insert(i);
            }
            set
        }));

        let orswot = Arc::new(Mutex::new({
            let mut set = Orswot::new();
            for i in 0..*size {
                let add_ctx = set.read_ctx().derive_add_ctx(i as u64);
                let add_op = set.add(i, add_ctx);
                set.apply(add_op);
            }
            set
        }));

        let gcounter = Arc::new(Mutex::new({
            let mut counter = GCounter::new();
            for _ in 0..*size {
                counter.inc(1);
            }
            counter
        }));

        let pncounter = Arc::new(Mutex::new({
            let mut counter = PNCounter::new();
            for i in 0..*size {
                if i % 2 == 0 {
                    counter.inc(1);
                } else {
                    counter.dec(1);
                }
            }
            counter
        }));

        let lwwreg = Arc::new(Mutex::new({
            let mut reg = LWWReg { val: 42, marker: 0 };
            for i in 0..*size {
                reg.update(i as u64, i as u64);
            }
            reg
        }));

        let mvreg = Arc::new(Mutex::new({
            let mut reg = MVReg::new();
            for i in 0..*size {
                let read_ctx = reg.read_ctx();
                let add_ctx = read_ctx.derive_add_ctx(i as u64);
                let op = reg.write(i, add_ctx);
                reg.apply(op);
            }
            reg
        }));

        let map = Arc::new(Mutex::new({
            let mut map: TestMap = Map::new();
            for i in 0..*size {
                let add_ctx = map
                    .read_ctx()
                    .derive_add_ctx((i as u64).try_into().unwrap());
                let op = map.update(i as u32, add_ctx, |v, ctx| v.write(i as u32, ctx));
                map.apply(op);
            }
            map
        }));

        let list = Arc::new(Mutex::new({
            let mut list = List::new();
            for i in 0..*size {
                let op = list.insert_index(i, i, i.to_string());
                list.apply(op);
            }
            list
        }));

        let glist = Arc::new(Mutex::new({
            let mut list = GList::new();
            for i in 0..*size {
                let op = list.insert_after(None, i);
                list.apply(op);
            }
            list
        }));

        // Custom benchmark for Concurrent GSet Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent GSet Multiple Reads {}", size),
            || {
                let gset = Arc::clone(&gset);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = gset.clone();
                                move || {
                                    let gset = value.lock().unwrap();
                                    for i in 0..*size {
                                        black_box(gset.contains(&i));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent ORSWOT Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent Orswot Multiple Reads {}", size),
            || {
                let orswot = Arc::clone(&orswot);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = orswot.clone();
                                move || {
                                    let orswot = value.lock().unwrap();
                                    for i in 0..*size {
                                        black_box(orswot.contains(&i));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent GCounter Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent GCounter Multiple Reads {}", size),
            || {
                let gcounter = Arc::clone(&gcounter);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = gcounter.clone();
                                move || {
                                    let gcounter = value.lock().unwrap();
                                    for _ in 0..*size {
                                        black_box(gcounter.read());
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent PNCounter Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent PNCounter Multiple Reads {}", size),
            || {
                let pncounter = Arc::clone(&pncounter);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = pncounter.clone();
                                move || {
                                    let pncounter = value.lock().unwrap();
                                    for _ in 0..*size {
                                        black_box(pncounter.read());
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent LWWReg Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent LWWReg Multiple Reads {}", size),
            || {
                let lwwreg = Arc::clone(&lwwreg);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = lwwreg.clone();
                                move || {
                                    let lwwreg = value.lock().unwrap();
                                    for _ in 0..*size {
                                        black_box(lwwreg.val);
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent MVReg Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent MVReg Multiple Reads {}", size),
            || {
                let mvreg = Arc::clone(&mvreg);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = mvreg.clone();
                                move || {
                                    let mvreg = value.lock().unwrap();
                                    for _ in 0..*size {
                                        black_box(mvreg.read());
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent Map Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent Map Multiple Reads {}", size),
            || {
                let map = Arc::clone(&map);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = map.clone();
                                move || {
                                    let map = value.lock().unwrap();
                                    for i in 0..*size {
                                        black_box(map.get(&i.try_into().unwrap()));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent List Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent List Multiple Reads {}", size),
            || {
                let list = Arc::clone(&list);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = list.clone();
                                move || {
                                    let list = value.lock().unwrap();
                                    for i in 0..*size {
                                        black_box(list.position(i));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );

        // Custom benchmark for Concurrent GList Multiple Reads
        custom_benchmark(
            &mut group,
            &format!("Concurrent GList Multiple Reads {}", size),
            || {
                let glist = Arc::clone(&glist);
                Box::new(move || {
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|_| {
                            thread::spawn({
                                let value = glist.clone();
                                move || {
                                    let glist = value.lock().unwrap();
                                    for i in 0..*size {
                                        black_box(glist.get(i));
                                    }
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

fn mixed_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Operations");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        custom_benchmark(
            &mut group,
            &format!("GSet Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut gset = GSet::new();
                    let mut rng = rand::thread_rng();
                    for _ in 0..*size {
                        match rng.gen_range(0..4) {
                            0 => gset.insert(black_box(rng.gen::<u32>())), // Insert
                            1 => {
                                if !gset.read().is_empty() {
                                    let value = gset.read().iter().next().unwrap().clone();
                                    black_box(gset.contains(&value)); // Read
                                }
                            }
                            _ => {} // GSet doesn't support remove or update
                        }
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Orswot Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut orswot = Orswot::new();
                    let mut rng = rand::thread_rng();
                    let mut inserted = HashSet::new();
                    for _ in 0..*size {
                        match rng.gen_range(0..4) {
                            0 => {
                                // Insert
                                let value = rng.gen::<u32>();
                                let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                                let op = orswot.add(black_box(value), add_ctx);
                                orswot.apply(op);
                                inserted.insert(value);
                            }
                            1 => {
                                // Remove
                                if let Some(&value) = inserted.iter().next() {
                                    let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                                    let op = orswot.rm(black_box(value), rm_ctx);
                                    orswot.apply(op);
                                    inserted.remove(&value);
                                }
                            }
                            2 => {
                                // Update (remove then insert)
                                if let Some(&value) = inserted.iter().next() {
                                    let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                                    let rm_op = orswot.rm(black_box(value), rm_ctx);
                                    orswot.apply(rm_op);
                                    inserted.remove(&value);

                                    let new_value = rng.gen::<u32>();
                                    let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                                    let add_op = orswot.add(black_box(new_value), add_ctx);
                                    orswot.apply(add_op);
                                    inserted.insert(new_value);
                                }
                            }
                            3 => {
                                // Read
                                if let Some(&value) = inserted.iter().next() {
                                    black_box(orswot.contains(&value));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GCounter Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut gcounter = GCounter::new();
                    let mut rng = rand::thread_rng();
                    for _ in 0..*size {
                        match rng.gen_range(0..2) {
                            0 => {
                                let op = gcounter.inc(black_box(rng.gen_range(1..10)));
                                gcounter.apply(op);
                            }
                            1 => {
                                black_box(gcounter.read());
                            }
                            _ => unreachable!(),
                        };
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("PNCounter Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut pncounter = PNCounter::new();
                    let mut rng = rand::thread_rng();
                    for _ in 0..*size {
                        match rng.gen_range(0..3) {
                            0 => {
                                let op = pncounter.inc(black_box(rng.gen_range(1..10)));
                                pncounter.apply(op);
                            }
                            1 => {
                                let op = pncounter.dec(black_box(rng.gen_range(1..10)));
                                pncounter.apply(op);
                            }
                            2 => {
                                black_box(pncounter.read());
                            }
                            _ => unreachable!(),
                        };
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("LWWReg Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut lww_reg = LWWReg { val: 0, marker: 0 };
                    let mut rng = rand::thread_rng();
                    for i in 0..*size {
                        match rng.gen_range(0..2) {
                            0 => {
                                lww_reg.update(black_box(i as u64), black_box(rng.gen::<u32>()));
                            }
                            1 => {
                                black_box(lww_reg.val);
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("MVReg Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut mv_reg = MVReg::new();
                    let mut rng = rand::thread_rng();
                    for _ in 0..*size {
                        match rng.gen_range(0..2) {
                            0 => {
                                // Write
                                let ctx = mv_reg.read_ctx().derive_add_ctx(1);
                                let op = mv_reg.write(black_box(rng.gen::<u32>()), ctx);
                                mv_reg.apply(op);
                            }
                            1 => {
                                black_box(mv_reg.read());
                            } // Read
                            _ => unreachable!(),
                        }
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Map Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut map: TestMap = Map::new();
                    let mut rng = rand::thread_rng();
                    let mut keys = Vec::new();
                    for _ in 0..*size {
                        match rng.gen_range(0..4) {
                            0 => {
                                // Insert
                                let key = rng.gen::<u32>();
                                let add_ctx = map.read_ctx().derive_add_ctx(1);
                                let op = map.update(black_box(key), add_ctx, |v, ctx| {
                                    v.write(black_box(rng.gen::<u32>().try_into().unwrap()), ctx)
                                });
                                map.apply(op);
                                keys.push(key);
                            }
                            1 => {
                                // Remove
                                if let Some(&key) = keys.last() {
                                    let rm_ctx = map.read_ctx().derive_rm_ctx();
                                    let op = map.rm(black_box(key), rm_ctx);
                                    map.apply(op);
                                    keys.pop();
                                }
                            }
                            2 => {
                                // Update
                                if let Some(&key) = keys.last() {
                                    let add_ctx = map.read_ctx().derive_add_ctx(1);
                                    let op = map.update(black_box(key), add_ctx, |v, ctx| {
                                        v.write(
                                            black_box(rng.gen::<u32>().try_into().unwrap()),
                                            ctx,
                                        )
                                    });
                                    map.apply(op);
                                }
                            }
                            3 => {
                                // Read
                                if let Some(&key) = keys.last() {
                                    black_box(map.get(&key));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("List Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut list = List::new();
                    let mut rng = rand::thread_rng();
                    for i in 0..*size {
                        match rng.gen_range(0..4) {
                            0 => {
                                // Insert
                                let op = list.insert_index(i, black_box(rng.gen::<u32>()), "actor");
                                list.apply(op);
                            }
                            1 => {
                                // Remove
                                if i > 0 {
                                    if let Some(op) = list.delete_index((i - 1).into(), "actor") {
                                        list.apply(op);
                                    }
                                }
                            }
                            2 => {
                                // Update
                                if i > 0 {
                                    if let Some(del_op) = list.delete_index((i - 1).into(), "actor")
                                    {
                                        list.apply(del_op);
                                    }
                                    let ins_op = list.insert_index(
                                        (i - 1).into(),
                                        black_box(rng.gen::<u32>()),
                                        "actor",
                                    );
                                    list.apply(ins_op);
                                }
                            }
                            3 => {
                                // Read
                                if i > 0 {
                                    black_box(list.position(i - 1));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GList Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mut glist = GList::new();
                    let mut rng = rand::thread_rng();
                    for _ in 0..*size {
                        match rng.gen_range(0..2) {
                            0 => {
                                // Insert
                                let op =
                                    glist.insert_after(glist.last(), black_box(rng.gen::<u32>()));
                                glist.apply(op);
                            }
                            1 => {
                                // Read
                                if let Some(index) = glist.len().checked_sub(1) {
                                    black_box(glist.get(index));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

fn concurrent_mixed_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Mixed Operations");
    const NUM_THREADS: usize = 4;
    let sizes = [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ];

    for size in [
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000,
    ]
    .iter()
    {
        custom_benchmark(
            &mut group,
            &format!("GSet Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let gset = Arc::new(RwLock::new(GSet::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let gset = Arc::clone(&gset);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..4) {
                                0 => gset.write().unwrap().insert(rng.gen::<u32>()),
                                1 => {
                                    let gset_read = gset.read().unwrap().read();
                                    if !gset_read.is_empty() {
                                        if let Some(&value) = gset_read.iter().next() {
                                            gset_read.contains(&value);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Orswot Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let orswot = Arc::new(RwLock::new(Orswot::new()));
                    let inserted = Arc::new(RwLock::new(HashSet::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let orswot = Arc::clone(&orswot);
                        let inserted = Arc::clone(&inserted);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..4) {
                                0 => {
                                    let value = rng.gen::<u32>();
                                    let mut orswot = orswot.write().unwrap();
                                    let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                                    let op = orswot.add(value, add_ctx);
                                    orswot.apply(op);
                                    inserted.write().unwrap().insert(value);
                                }
                                1 => {
                                    let mut inserted = inserted.write().unwrap();
                                    if let Some(&value) = inserted.iter().next() {
                                        let mut orswot = orswot.write().unwrap();
                                        let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                                        let op = orswot.rm(value, rm_ctx);
                                        orswot.apply(op);
                                        inserted.remove(&value);
                                    }
                                }
                                2 => {
                                    let mut inserted = inserted.write().unwrap();
                                    if let Some(&value) = inserted.iter().next() {
                                        let mut orswot = orswot.write().unwrap();
                                        let rm_ctx = orswot.read_ctx().derive_rm_ctx();
                                        let rm_op = orswot.rm(value, rm_ctx);
                                        orswot.apply(rm_op);
                                        inserted.remove(&value);

                                        let new_value = rng.gen::<u32>();
                                        let add_ctx = orswot.read_ctx().derive_add_ctx(1);
                                        let add_op = orswot.add(new_value, add_ctx);
                                        orswot.apply(add_op);
                                        inserted.insert(new_value);
                                    }
                                }
                                3 => {
                                    let inserted = inserted.read().unwrap();
                                    if let Some(&value) = inserted.iter().next() {
                                        let orswot = orswot.read().unwrap();
                                        orswot.contains(&value);
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GCounter Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let gcounter = Arc::new(RwLock::new(GCounter::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let gcounter = Arc::clone(&gcounter);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..2) {
                                0 => {
                                    let mut gcounter = gcounter.write().unwrap();
                                    let op = gcounter.inc(rng.gen_range(1..10));
                                    gcounter.apply(op);
                                }
                                1 => {
                                    let gcounter = gcounter.read().unwrap();
                                    gcounter.read();
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("PNCounter Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let pncounter = Arc::new(RwLock::new(PNCounter::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let pncounter = Arc::clone(&pncounter);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..3) {
                                0 => {
                                    let mut pncounter = pncounter.write().unwrap();
                                    let op = pncounter.inc(rng.gen_range(1..10));
                                    pncounter.apply(op);
                                }
                                1 => {
                                    let mut pncounter = pncounter.write().unwrap();
                                    let op = pncounter.dec(rng.gen_range(1..10));
                                    pncounter.apply(op);
                                }
                                2 => {
                                    let pncounter = pncounter.read().unwrap();
                                    pncounter.read();
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("LWWReg Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let lww_reg = Arc::new(RwLock::new(LWWReg { val: 0, marker: 0 }));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let lww_reg = Arc::clone(&lww_reg);
                        let mut rng = rand::thread_rng();
                        for i in 0..size / NUM_THREADS {
                            match rng.gen_range(0..2) {
                                0 => {
                                    let mut lww_reg = lww_reg.write().unwrap();
                                    lww_reg.update(i as u64, rng.gen::<u32>());
                                }
                                1 => {
                                    let lww_reg = lww_reg.read().unwrap();
                                    lww_reg.val;
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("MVReg Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let mv_reg = Arc::new(RwLock::new(MVReg::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let mv_reg = Arc::clone(&mv_reg);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..2) {
                                0 => {
                                    let mut mv_reg = mv_reg.write().unwrap();
                                    let ctx = mv_reg.read_ctx().derive_add_ctx(1);
                                    let op = mv_reg.write(rng.gen::<u32>(), ctx);
                                    mv_reg.apply(op);
                                }
                                1 => {
                                    let mv_reg = mv_reg.read().unwrap();
                                    mv_reg.read();
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("Map Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let map: Arc<RwLock<TestMap>> = Arc::new(RwLock::new(Map::new()));
                    let keys = Arc::new(RwLock::new(Vec::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..4) {
                                0 => {
                                    let key = rng.gen::<u32>();
                                    let mut map = map.write().unwrap();
                                    let add_ctx = map.read_ctx().derive_add_ctx(1);
                                    let op = map.update(key, add_ctx, |v, ctx| {
                                        v.write(rng.gen::<u32>().try_into().unwrap(), ctx)
                                    });
                                    map.apply(op);
                                    keys.write().unwrap().push(key);
                                }
                                1 => {
                                    let mut keys = keys.write().unwrap();
                                    if let Some(&key) = keys.last() {
                                        let mut map = map.write().unwrap();
                                        let rm_ctx = map.read_ctx().derive_rm_ctx();
                                        let op = map.rm(key, rm_ctx);
                                        map.apply(op);
                                        keys.pop();
                                    }
                                }
                                2 => {
                                    let keys = keys.read().unwrap();
                                    if let Some(&key) = keys.last() {
                                        let mut map = map.write().unwrap();
                                        let add_ctx = map.read_ctx().derive_add_ctx(1);
                                        let op = map.update(key, add_ctx, |v, ctx| {
                                            v.write(rng.gen::<u32>().try_into().unwrap(), ctx)
                                        });
                                        map.apply(op);
                                    }
                                }
                                3 => {
                                    let keys = keys.read().unwrap();
                                    if let Some(&key) = keys.last() {
                                        let map = map.read().unwrap();
                                        map.get(&key);
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("List Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let list = Arc::new(RwLock::new(List::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let list = Arc::clone(&list);
                        let mut rng = rand::thread_rng();
                        for i in 0..size / NUM_THREADS {
                            match rng.gen_range(0..4) {
                                0 => {
                                    let mut list = list.write().unwrap();
                                    let op = list.insert_index(i, rng.gen::<u32>(), "actor");
                                    list.apply(op);
                                }
                                1 => {
                                    if i > 0 {
                                        let mut list = list.write().unwrap();
                                        if let Some(op) = list.delete_index((i - 1).into(), "actor")
                                        {
                                            list.apply(op);
                                        }
                                    }
                                }
                                2 => {
                                    if i > 0 {
                                        let mut list = list.write().unwrap();
                                        if let Some(del_op) =
                                            list.delete_index((i - 1).into(), "actor")
                                        {
                                            list.apply(del_op);
                                        }
                                        let ins_op = list.insert_index(
                                            (i - 1).into(),
                                            rng.gen::<u32>(),
                                            "actor",
                                        );
                                        list.apply(ins_op);
                                    }
                                }
                                3 => {
                                    if i > 0 {
                                        let list = list.read().unwrap();
                                        list.position(i - 1);
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );

        custom_benchmark(
            &mut group,
            &format!("GList Concurrent Mixed Operations {}", size),
            || {
                Box::new(move || {
                    let glist = Arc::new(RwLock::new(GList::new()));
                    (0..NUM_THREADS).into_iter().for_each(|_| {
                        let glist = Arc::clone(&glist);
                        let mut rng = rand::thread_rng();
                        for _ in 0..size / NUM_THREADS {
                            match rng.gen_range(0..2) {
                                0 => {
                                    let mut glist = glist.write().unwrap();
                                    let op = glist.insert_after(glist.last(), rng.gen::<u32>());
                                    glist.apply(op);
                                }
                                1 => {
                                    let glist = glist.read().unwrap();
                                    if let Some(index) = glist.len().checked_sub(1) {
                                        glist.get(index);
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    });
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    // insert_single_benchmark,
    insert_multiple_benchmark,
    concurrent_insert_single_benchmark,
    concurrent_insert_multiple_benchmark,
    update_single_benchmark,
    update_multiple_benchmark,
    concurrent_update_single_benchmark,
    concurrent_update_multiple_benchmark,
    remove_single_benchmark,
    remove_multiple_benchmark,
    concurrent_remove_single_benchmark,
    concurrent_remove_multiple_benchmark,
    merge_single_benchmark,
    merge_multiple_benchmark,
    concurrent_merge_single_benchmark,
    concurrent_merge_multiple_benchmark,
    read_single_benchmark,
    read_multiple_benchmark,
    concurrent_read_single_benchmark,
    concurrent_read_multiple_benchmark,
    concurrent_remove_multiple_benchmark,
    mixed_operations_benchmark,
    concurrent_mixed_operations_benchmark,
);
criterion_main!(benches);
