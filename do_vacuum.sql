.echo on
PRAGMA synchronous = OFF;
PRAGMA journal_mode = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;
vacuum;

