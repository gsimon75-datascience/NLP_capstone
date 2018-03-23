#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include <sqlite3.h>
#include <new>

#define INPUT_FILENAME "sample.normalised.sentences.bin"
#define N_max_prefix_size  4

#define END_SENTENCE 0xffffffff

uint32_t *input;
int input_len;

// NOTE: size is on the scale of malloc chunk overhead, so
// it makes little sense to allocate object and manage its ptr

struct follower_t {
    uint32_t word;
    uint32_t occurences;

    follower_t(uint32_t n_word) : word(n_word), occurences(1) { };
};

struct prefix_t {
    uint32_t word;
    uint32_t occurences;
    uint32_t num_children;
    prefix_t *children;
    uint32_t num_followers;
    follower_t *followers;

    prefix_t(uint32_t n_word);
    void book_follower(uint32_t follower_word);
    prefix_t *get_child(uint32_t child_word, uint32_t follower_word);
    ~prefix_t();
    //void dump(int indent) const;
    //void dump_children() const;
    //void dump_followers() const;
};

prefix_t root(-1);

prefix_t::prefix_t(uint32_t n_word)
        : word(n_word), occurences(0), num_children(0), children(nullptr), num_followers(0), followers(nullptr) { }

prefix_t::~prefix_t() {
    free(followers);
    for (uint32_t i = 0; i < num_children; i++)
        children[i].~prefix_t();
    free(children); 
}

/*void prefix_t::dump_followers() const {
    printf("(%d)[", num_followers);
    for (uint32_t i = 0; i < num_followers; i++) {
        if (i)
            printf(", ");
        printf("(%d)%d", followers[i].occurences, followers[i].word);
    }
    printf("]");
}*/

/*void prefix_t::dump_children() const {
    printf("(%d)[", num_children);
    for (uint32_t i = 0; i < num_children; i++) {
        if (i)
            printf(", ");
        printf("%d", children[i].word);
    }
    printf("]");
}*/

/*void prefix_t::dump(int indent) const {
    for (int i = 0; i < indent; i++)
        printf("  ");
    printf("word=%d, n=%d, followers=", word, occurences); dump_followers(); printf("\n");
    for (uint32_t i = 0; i < num_children; i++)
        children[i].dump(indent + 1);
}*/


void
prefix_t::book_follower(uint32_t follower_word) {
    //printf("  Searching for follower %d in ", follower_word); dump_followers();
    // find where it should be
    uint32_t min = 0, max = num_followers, middle = 0;
    while (min < max) {
        middle = (min + max) / 2;
        if (followers[middle].word < follower_word)
            min = middle + 1;
        else if (followers[middle].word > follower_word)
            max = middle;
        else {
            //printf(": FOUND at %d\n", middle);
            followers[middle].occurences++;
            return;
        }
    }
    //printf(": CREATING NEW at %d\n", min);
    //printf("    Followers before insertion: "); dump_followers(); printf("\n");
    // min = max = the position where it should be
    followers = (follower_t*)realloc(followers, (1 + num_followers) * sizeof(follower_t));
    if (num_followers > min)
        memmove(&followers[min + 1], &followers[min], (num_followers - min) * sizeof(follower_t));
    num_followers++;
    new(&followers[min]) follower_t(follower_word);
    //printf("    Followers after  insertion: "); dump_followers(); printf("\n");
}

prefix_t* prefix_t::get_child(uint32_t child_word, uint32_t follower_word) {
    // find where it should be
    uint32_t min = 0, max = num_children, middle = 0;
    //printf("Searching for child %d of %d in ", child_word, word); dump_children();
    while (min < max) {
        middle = (min + max) / 2;
        if (children[middle].word < child_word)
            min = middle + 1;
        else if (children[middle].word > child_word)
            max = middle;
        else {
            //printf(": FOUND at %d\n", middle);
            return &children[middle];
        }
    }

    //printf(": CREATING NEW at %d\n", min);
    //printf("  Children before insertion: "); dump_children(); printf("\n");
    // min = max = the position where it should be
    children = (prefix_t*)realloc(children, (1 + num_children) * sizeof(prefix_t));
    if (num_children > min)
        memmove(&children[min + 1], &children[min], (num_children - min) * sizeof(prefix_t));
    num_children++;
    new(&children[min]) prefix_t(child_word);
    //printf("  Children after  insertion: "); dump_children(); printf("\n");
    return &children[min];
}


void
generate_ngrams(void) {
    uint32_t *after_last = &(input[input_len]);
    uint32_t *end;

    for (uint32_t *start = input; start < after_last; start = end + 1) {
        //printf("-- Start of line\n");
        for (end = start; *end != END_SENTENCE; ++end)
            ;
        for (uint32_t *follower = end - 1; (follower > start); --follower) {
            uint32_t *pfx_start = follower - 1;
            prefix_t *p = &root;
            for (int i = 0; (i < N_max_prefix_size) && (pfx_start >= start); ++i, --pfx_start) {
                /*printf("-- Adding prefix=[");
                for (uint32_t *pp = pfx_start; pp < follower; ++pp) {
                    if (pp != pfx_start)
                        printf(", ");
                    printf("%d", *pp);
                }
                printf("], follower=%d\n", *follower);*/

                p = p->get_child(*pfx_start, *follower);
                p->occurences++;
                p->book_follower(*follower);

                //printf("-- Result:\n"); root.dump(0);

            }
        }
        //printf("-- End of line\n");
    }
}

int
main(void) {
    int fd_in = open(INPUT_FILENAME, O_RDONLY);
    if (fd_in < 0) {
        fprintf(stderr, "Open file failed: (%d) %s\n", errno, strerror(errno));
        return -1;
    }

    struct stat st;
    fstat(fd_in, &st);
    input_len = st.st_size / 4;

    void *p = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd_in, 0);
    if (p == (void*)-1) {
        fprintf(stderr, "Mmap failed: (%d) %s\n", errno, strerror(errno));
        return -2;
    }
    input = (uint32_t*)p;

    generate_ngrams();

    munmap(p, st.st_size);
    close(fd_in);
    return 0;
}
