#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>

#include <sqlite3.h>
#include <new>

//#define INPUT_FILENAME "sample.normalised.sentences.bin"
//#define DB_FILENAME "test.db"

#define INPUT_FILENAME "final/en_US/all.normalised.sentences.bin"
#define DB_FILENAME "dict.db"

#define N_max_prefix_size  4

#define END_SENTENCE -1

// NOTE: size is on the scale of malloc chunk overhead, so
// it makes little sense to allocate object and manage its ptr

// NOTE: I've removed a lot of debugging code, if you need them, check this delta

struct follower_t {
    int32_t word;
    int32_t occurences;

    follower_t(int32_t n_word) : word(n_word), occurences(1) { };
};

struct prefix_t {
    int32_t word;
    int32_t occurences;
    int32_t num_children;
    prefix_t *children;
    int32_t num_followers;
    follower_t *followers;

    prefix_t(int32_t n_word);
    void book_follower(int32_t follower_word);
    prefix_t *get_child(int32_t child_word, int32_t follower_word);
    void commit(int32_t parent_id) const;
    ~prefix_t();
};

sqlite3 *db = nullptr;
sqlite3_stmt *qNewPrefix = nullptr;
sqlite3_stmt *qNewNgram = nullptr;
prefix_t root(-1);

long total_lines = 0;
long total_prefixes = 0;
long total_ngrams = 0;

bool
need_progress_printout() {
    static time_t last_progress_printout = time(nullptr);
    time_t now = time(nullptr);
    if ((now - last_progress_printout) > 0) {
        last_progress_printout = now;
        return true;
    }
    return false;
}

const char *
strnow() {
    static char timestr_now[128];
    struct tm tm_now;
    time_t now;

    time(&now);
    strftime(timestr_now, sizeof(timestr_now), "%T", localtime_r(&now, &tm_now));
    return timestr_now;
}

prefix_t::prefix_t(int32_t n_word)
        : word(n_word), occurences(0), num_children(0), children(nullptr), num_followers(0), followers(nullptr) { }

prefix_t::~prefix_t() {
    free(followers);
    for (int32_t i = 0; i < num_children; i++)
        children[i].~prefix_t();
    free(children);
}

void
prefix_t::book_follower(int32_t follower_word) {
    // find where it should be
    int32_t min = 0, max = num_followers, middle = 0;
    while (min < max) {
        middle = (min + max) / 2;
        if (followers[middle].word < follower_word)
            min = middle + 1;
        else if (followers[middle].word > follower_word)
            max = middle;
        else {
            followers[middle].occurences++;
            return;
        }
    }
    total_ngrams++;
    // min = max = the position where it should be
    followers = (follower_t*)realloc(followers, (1 + num_followers) * sizeof(follower_t));
    if (num_followers > min)
        memmove(&followers[min + 1], &followers[min], (num_followers - min) * sizeof(follower_t));
    num_followers++;
    new(&followers[min]) follower_t(follower_word);
}

prefix_t* prefix_t::get_child(int32_t child_word, int32_t follower_word) {
    // find where it should be
    int32_t min = 0, max = num_children, middle = 0;
    while (min < max) {
        middle = (min + max) / 2;
        if (children[middle].word < child_word)
            min = middle + 1;
        else if (children[middle].word > child_word)
            max = middle;
        else {
            return &children[middle];
        }
    }
    total_prefixes++;
    // min = max = the position where it should be
    children = (prefix_t*)realloc(children, (1 + num_children) * sizeof(prefix_t));
    if (num_children > min)
        memmove(&children[min + 1], &children[min], (num_children - min) * sizeof(prefix_t));
    num_children++;
    new(&children[min]) prefix_t(child_word);
    return &children[min];
}


void
generate_ngrams(int32_t *input, int32_t *after_last) {
    int32_t *end;

    total_lines = total_prefixes = total_ngrams = 0;
    for (int32_t *start = input; start < after_last; start = end + 1) {
        for (end = start; *end != END_SENTENCE; ++end)
            ;
        for (int32_t *follower = end - 1; (follower > start); --follower) {
            int32_t *pfx_start = follower - 1;
            prefix_t *p = &root;
            for (int i = 0; (i < N_max_prefix_size) && (pfx_start >= start); ++i, --pfx_start) {
                p = p->get_child(*pfx_start, *follower);
                p->occurences++;
                p->book_follower(*follower);
            }
        }
        total_lines++;

        if (need_progress_printout())
            printf("%s Processed; lines=%ld, prefixes=%ld, ngrams=%ld\n", strnow(), total_lines, total_prefixes, total_ngrams);
    }
}

int32_t prefix_id = 0;

void
prefix_t::commit(int32_t parent_id) const {
    if (word == -1) {
        for (int32_t i = 0; i < num_children; i++)
            children[i].commit(-1);
    }
    else if (occurences > 1) {
        int res, valid_followers = 0;
        int32_t id = prefix_id++;

        sqlite3_bind_int(qNewNgram, 1, id);
        for (int32_t i = 0; i < num_followers; i++) {
            if (followers[i].occurences > 1) {
                sqlite3_bind_int(qNewNgram, 2, followers[i].word);
                sqlite3_bind_int(qNewNgram, 3, followers[i].occurences);
                res = sqlite3_step(qNewNgram);
                if (res != SQLITE_DONE)
                    printf("%s ERROR: Cannot insert ngram: %d\n", strnow(), res);
                sqlite3_reset(qNewNgram);
                total_ngrams++;
                valid_followers++;
            }
        }

        if (valid_followers > 0) {
            sqlite3_bind_int(qNewPrefix, 1, id);
            sqlite3_bind_int(qNewPrefix, 2, parent_id);
            sqlite3_bind_int(qNewPrefix, 3, word);
            sqlite3_bind_int(qNewPrefix, 4, occurences);
            res = sqlite3_step(qNewPrefix);
            if (res != SQLITE_DONE)
                printf("%s ERROR: Cannot insert prefix: %d\n",strnow(),  res);
            sqlite3_reset(qNewPrefix);
            total_prefixes++;

            if (need_progress_printout())
                printf("%s Committing; prefixes=%ld, ngrams=%ld\n", strnow(), total_prefixes, total_ngrams);

            for (int32_t i = 0; i < num_children; i++)
                children[i].commit(id);
        }
    }
}

int
main(void) {
    int res;

    int fd_in = open(INPUT_FILENAME, O_RDONLY);
    if (fd_in < 0) {
        fprintf(stderr, "Cannot open input file %s: (%d) %s\n", INPUT_FILENAME, errno, strerror(errno));
        return -1;
    }

    struct stat st;
    res = fstat(fd_in, &st);
    if (res < 0) {
        fprintf(stderr, "Cannot stat input file: (%d) %s\n", errno, strerror(errno));
        return -1;
    }

    int input_len = st.st_size / 4;

    void *p = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd_in, 0);
    if (p == (void*)-1) {
        fprintf(stderr, "Mmap failed: (%d) %s\n", errno, strerror(errno));
        return -1;
    }
    int32_t *input = (int32_t*)p;
    int32_t *after_last = &(input[input_len]);

    res = sqlite3_open(DB_FILENAME, &db);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Cannot open database %s: %d\n", DB_FILENAME, res);
        return -2;
    }

    res = sqlite3_exec(db,
        "PRAGMA synchronous = OFF;"
        "PRAGMA secure_delete = OFF;"
        "PRAGMA locking_mode = EXCLUSIVE;"
        "PRAGMA mmap_size = 4294967296;"
        "PRAGMA threads = 2;"
        "BEGIN", nullptr, nullptr, nullptr);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Cannot set pragmas: %d\n", res);
            return -2;
    }

    res = sqlite3_prepare_v2(db, "INSERT INTO prefix_t (id, parent, word, occurences) VALUES (?1, ?2, ?3, ?4)", -1, &qNewPrefix, NULL);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Cannot prepare prefix-insert query: %d\n", res);
            return -2;
    }

    res = sqlite3_prepare_v2(db, "INSERT INTO ngram_t (prefix, follower, occurences) VALUES (?1, ?2, ?3)", -1, &qNewNgram, NULL);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Cannot prepare prefix-insert query: %d\n", res);
            return -2;
    }

    printf("%s Collecting N-grams\n", strnow());
    generate_ngrams(input, after_last);

    munmap(p, st.st_size);
    close(fd_in);

    printf("%s Committing N-grams\n", strnow());
    total_prefixes = total_ngrams = 0;
    root.commit(-1);

    res = sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Cannot commit changes: %d\n", res);
            return -2;
    }

    sqlite3_finalize(qNewNgram);
    sqlite3_finalize(qNewPrefix);

    res = sqlite3_close(db);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Cannot close the database: %d\n", res);
            return -2;
    }

    printf("%s Done. (Please wait for the process to finish on its own.)\n", strnow());
    return 0;
}
