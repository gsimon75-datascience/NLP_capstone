.echo on
PRAGMA synchronous = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;
PRAGMA cache_size=20000;

vacuum;

