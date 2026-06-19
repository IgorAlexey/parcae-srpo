#!/usr/bin/env python3
"""Build NL-spec -> C -> x86-64 asm triples for SRPO recurrent-depth training.

Generates ~200 correct, hand-written C implementations for algorithmic
functions, compiles each with GCC -O2 to x86-64 assembly (Intel syntax),
verifies the assembly compiles back, and saves both raw triples and
SRPO-format training records.

Usage:
    python scripts/build_nl_asm_dataset.py                    # generate all
    python scripts/build_nl_asm_dataset.py --dry-run          # validate only
    python scripts/build_nl_asm_dataset.py --start 0 --count 20  # first 20
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ── 200 NL spec → C implementation pairs ──────────────────────────

SPECS: list[dict] = []


def _s(spec: str, c_code: str) -> None:
    SPECS.append({"spec": spec, "c_code": c_code.strip()})


# ── Bit manipulation (30) ──────────────────────────────────────────

_s(
    "Write a C function that counts set bits in a 64-bit unsigned integer using Brian Kernighan's algorithm.",
    """#include <stdint.h>
int popcount_u64(uint64_t n) {
    int c = 0;
    while (n) { n &= n - 1; c++; }
    return c;
}""",
)

_s(
    "Write a C function that counts set bits in a 32-bit unsigned integer using a parallel bit-count approach.",
    """#include <stdint.h>
int popcount_u32(uint32_t n) {
    n = n - ((n >> 1) & 0x55555555u);
    n = (n & 0x33333333u) + ((n >> 2) & 0x33333333u);
    n = (n + (n >> 4)) & 0x0F0F0F0Fu;
    return (int)((n * 0x01010101u) >> 24);
}""",
)

_s(
    "Write a C function that reverses the bits of a 32-bit unsigned integer.",
    """#include <stdint.h>
uint32_t reverse_bits_u32(uint32_t n) {
    n = ((n & 0x55555555u) << 1) | ((n >> 1) & 0x55555555u);
    n = ((n & 0x33333333u) << 2) | ((n >> 2) & 0x33333333u);
    n = ((n & 0x0F0F0F0Fu) << 4) | ((n >> 4) & 0x0F0F0F0Fu);
    n = (n << 24) | ((n & 0xFF00u) << 8) | ((n >> 8) & 0xFF00u) | (n >> 24);
    return n;
}""",
)

_s(
    "Write a C function that returns the position of the most significant set bit (floor(log2)) of a 64-bit unsigned integer. Returns -1 for n=0.",
    """#include <stdint.h>
int msb_u64(uint64_t n) {
    if (n == 0) return -1;
    int pos = 0;
    if (n >> 32) { n >>= 32; pos += 32; }
    if (n >> 16) { n >>= 16; pos += 16; }
    if (n >> 8)  { n >>= 8;  pos += 8;  }
    if (n >> 4)  { n >>= 4;  pos += 4;  }
    if (n >> 2)  { n >>= 2;  pos += 2;  }
    if (n >> 1)  { pos += 1; }
    return pos;
}""",
)

_s(
    "Write a C function that returns the position of the least significant set bit (ctz) of a 64-bit unsigned integer using De Bruijn multiplication. Returns -1 for n=0.",
    """#include <stdint.h>
int lsb_u64(uint64_t n) {
    if (n == 0) return -1;
    static const int tab[64] = {
        0, 1, 2, 7, 3, 13, 8, 19, 4, 25, 14, 28, 9, 34, 20, 40,
        5, 17, 26, 38, 15, 46, 29, 48, 10, 31, 35, 54, 21, 50, 41, 57,
        63, 6, 12, 18, 24, 27, 33, 39, 16, 37, 45, 47, 30, 53, 49, 56,
        62, 11, 23, 32, 36, 44, 52, 55, 61, 22, 43, 51, 60, 42, 59, 58
    };
    return tab[(int)(((n & (~n + 1)) * 0x07EDD5E59A4E28C2ull) >> 58)];
}""",
)

_s(
    "Write a C function that rotates a 64-bit unsigned integer left by k bits.",
    """#include <stdint.h>
uint64_t rotl_u64(uint64_t n, int k) {
    k &= 63;
    return (n << k) | (n >> (64 - k));
}""",
)

_s(
    "Write a C function that rotates a 32-bit unsigned integer right by k bits.",
    """#include <stdint.h>
uint32_t rotr_u32(uint32_t n, int k) {
    k &= 31;
    return (n >> k) | (n << (32 - k));
}""",
)

_s(
    "Write a C function that swaps the even and odd bits of a 32-bit unsigned integer.",
    """#include <stdint.h>
uint32_t swap_even_odd_bits(uint32_t n) {
    return ((n & 0xAAAAAAAAu) >> 1) | ((n & 0x55555555u) << 1);
}""",
)

_s(
    "Write a C function that checks if a 32-bit unsigned integer is a power of two.",
    """#include <stdint.h>
#include <stdbool.h>
bool is_power_of_two(uint32_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}""",
)

_s(
    "Write a C function that rounds a 32-bit unsigned integer up to the next power of two.",
    """#include <stdint.h>
uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1; n |= n >> 2;
    n |= n >> 4; n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}""",
)

_s(
    "Write a C function that computes the parity (1 if odd number of set bits, 0 otherwise) of a 64-bit unsigned integer.",
    """#include <stdint.h>
int parity_u64(uint64_t n) {
    n ^= n >> 32;
    n ^= n >> 16;
    n ^= n >> 8;
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    return (int)(n & 1);
}""",
)

_s(
    "Write a C function that returns the number of leading zeros in a 32-bit unsigned integer.",
    """#include <stdint.h>
int clz_u32(uint32_t n) {
    if (n == 0) return 32;
    int c = 0;
    if ((n & 0xFFFF0000u) == 0) { c += 16; n <<= 16; }
    if ((n & 0xFF000000u) == 0) { c += 8;  n <<= 8;  }
    if ((n & 0xF0000000u) == 0) { c += 4;  n <<= 4;  }
    if ((n & 0xC0000000u) == 0) { c += 2;  n <<= 2;  }
    if ((n & 0x80000000u) == 0) { c += 1; }
    return c;
}""",
)

_s(
    "Write a C function that returns the number of trailing zeros in a 64-bit unsigned integer using De Bruijn sequences.",
    """#include <stdint.h>
int ctz_u64(uint64_t n) {
    if (n == 0) return 64;
    static const int tab[64] = {
        0, 1, 2, 7, 3, 13, 8, 19, 4, 25, 14, 28, 9, 34, 20, 40,
        5, 17, 26, 38, 15, 46, 29, 48, 10, 31, 35, 54, 21, 50, 41, 57,
        63, 6, 12, 18, 24, 27, 33, 39, 16, 37, 45, 47, 30, 53, 49, 56,
        62, 11, 23, 32, 36, 44, 52, 55, 61, 22, 43, 51, 60, 42, 59, 58
    };
    return tab[(int)(((n & (~n + 1)) * 0x07EDD5E59A4E28C2ull) >> 58)];
}""",
)

_s(
    "Write a C function that interleaves the low 16 bits of two 32-bit integers into one 32-bit integer (Morton Z-order).",
    """#include <stdint.h>
uint32_t interleave_u32(uint32_t x, uint32_t y) {
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    y = (y | (y << 8)) & 0x00FF00FFu;
    y = (y | (y << 4)) & 0x0F0F0F0Fu;
    y = (y | (y << 2)) & 0x33333333u;
    y = (y | (y << 1)) & 0x55555555u;
    return x | (y << 1);
}""",
)

_s(
    "Write a C function that returns the smallest power of two greater than or equal to a 64-bit unsigned integer.",
    """#include <stdint.h>
uint64_t ceil_pow2_u64(uint64_t n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;  n |= n >> 2;
    n |= n >> 4;  n |= n >> 8;
    n |= n >> 16; n |= n >> 32;
    return n + 1;
}""",
)

_s(
    "Write a C function that sets the k-th bit of a 32-bit unsigned integer.",
    """#include <stdint.h>
uint32_t set_bit(uint32_t n, int k) {
    return n | (1u << k);
}""",
)

_s(
    "Write a C function that clears the k-th bit of a 32-bit unsigned integer.",
    """#include <stdint.h>
uint32_t clear_bit(uint32_t n, int k) {
    return n & ~(1u << k);
}""",
)

_s(
    "Write a C function that toggles the k-th bit of a 32-bit unsigned integer.",
    """#include <stdint.h>
uint32_t toggle_bit(uint32_t n, int k) {
    return n ^ (1u << k);
}""",
)

_s(
    "Write a C function that returns the k-th bit of a 32-bit unsigned integer as a boolean.",
    """#include <stdint.h>
#include <stdbool.h>
bool test_bit(uint32_t n, int k) {
    return (n >> k) & 1u;
}""",
)

_s(
    "Write a C function that computes the bitwise AND of all numbers in a non-empty unsigned integer array.",
    """#include <stdint.h>
uint32_t bitwise_and_range(const uint32_t *arr, int len) {
    uint32_t result = arr[0];
    for (int i = 1; i < len; i++) result &= arr[i];
    return result;
}""",
)

_s(
    "Write a C function that computes the bitwise OR of all numbers in a non-empty unsigned integer array.",
    """#include <stdint.h>
uint32_t bitwise_or_range(const uint32_t *arr, int len) {
    uint32_t result = arr[0];
    for (int i = 1; i < len; i++) result |= arr[i];
    return result;
}""",
)

_s(
    "Write a C function that computes the bitwise XOR of all numbers in an integer array.",
    """int xor_range(const int *arr, int len) {
    int result = arr[0];
    for (int i = 1; i < len; i++) result ^= arr[i];
    return result;
}""",
)

_s(
    "Write a C function that checks if two 32-bit integers have opposite signs.",
    """#include <stdint.h>
#include <stdbool.h>
bool opposite_signs(int32_t a, int32_t b) {
    return (a ^ b) < 0;
}""",
)

_s(
    "Write a C function that computes the absolute value of a 32-bit integer without branching.",
    """int abs_no_branch(int n) {
    int mask = n >> 31;
    return (n + mask) ^ mask;
}""",
)

_s(
    "Write a C function that returns the sign of a 32-bit integer: -1, 0, or 1.",
    """int sign_i32(int n) {
    return (n > 0) - (n < 0);
}""",
)

_s(
    "Write a C function that swaps two integers using XOR (in-place via pointers).",
    """void xor_swap(int *a, int *b) {
    if (a != b) {
        *a ^= *b;
        *b ^= *a;
        *a ^= *b;
    }
}""",
)

_s(
    "Write a C function that multiplies a 32-bit integer by 7 without using the multiplication operator.",
    """int mul7(int n) {
    return (n << 3) - n;
}""",
)

_s(
    "Write a C function that divides a 32-bit unsigned integer by 8 without using the division operator.",
    """#include <stdint.h>
uint32_t div8(uint32_t n) {
    return n >> 3;
}""",
)

_s(
    "Write a C function that computes n modulo 8 for an unsigned 32-bit integer without using the modulo operator.",
    """#include <stdint.h>
uint32_t mod8(uint32_t n) {
    return n & 7u;
}""",
)

_s(
    "Write a C function that checks if an unsigned 32-bit integer is a multiple of 8.",
    """#include <stdint.h>
#include <stdbool.h>
bool is_multiple_of_8(uint32_t n) {
    return (n & 7u) == 0;
}""",
)

# ── String operations (25) ─────────────────────────────────────────

_s(
    "Write a C function that computes the length of a null-terminated string (strlen).",
    """#include <stddef.h>
size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return (size_t)(p - s);
}""",
)

_s(
    "Write a C function that copies a null-terminated source string into a destination buffer (strcpy), returning the destination pointer.",
    """char *my_strcpy(char *dst, const char *src) {
    char *d = dst;
    while ((*d++ = *src++));
    return dst;
}""",
)

_s(
    "Write a C function that copies at most n characters from source to destination (strncpy), padding with null bytes if needed.",
    """#include <stddef.h>
char *my_strncpy(char *dst, const char *src, size_t n) {
    size_t i;
    for (i = 0; i < n && src[i]; i++) dst[i] = src[i];
    for (; i < n; i++) dst[i] = '\\0';
    return dst;
}""",
)

_s(
    "Write a C function that concatenates source string to the end of destination string (strcat).",
    """char *my_strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d) d++;
    while ((*d++ = *src++));
    return dst;
}""",
)

_s(
    "Write a C function that compares two null-terminated strings lexicographically (strcmp).",
    """int my_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}""",
)

_s(
    "Write a C function that compares at most n characters of two strings (strncmp).",
    """#include <stddef.h>
int my_strncmp(const char *a, const char *b, size_t n) {
    while (n && *a && *a == *b) { n--; a++; b++; }
    return n ? (unsigned char)*a - (unsigned char)*b : 0;
}""",
)

_s(
    "Write a C function that locates the first occurrence of character c in string s (strchr).",
    """char *my_strchr(const char *s, int c) {
    while (*s && *s != (char)c) s++;
    return (*s == (char)c) ? (char *)s : 0;
}""",
)

_s(
    "Write a C function that locates the last occurrence of character c in string s (strrchr).",
    """char *my_strrchr(const char *s, int c) {
    const char *last = 0;
    do { if (*s == (char)c) last = s; } while (*s++);
    return (char *)last;
}""",
)

_s(
    "Write a C function that finds the first occurrence of the substring needle in haystack (strstr).",
    """#include <stddef.h>
char *my_strstr(const char *haystack, const char *needle) {
    if (!*needle) return (char *)haystack;
    for (; *haystack; haystack++) {
        if (*haystack == *needle) {
            size_t i;
            for (i = 0; needle[i] && haystack[i] == needle[i]; i++);
            if (!needle[i]) return (char *)haystack;
        }
    }
    return 0;
}""",
)

_s(
    "Write a C function that computes the length of the longest common prefix of two strings, bounded by n.",
    """#include <stddef.h>
size_t common_prefix_len(const char *a, const char *b, size_t n) {
    size_t i = 0;
    while (i < n && a[i] && a[i] == b[i]) i++;
    return i;
}""",
)

_s(
    "Write a C function that reverses a null-terminated string in-place.",
    """#include <stddef.h>
void reverse_str(char *s) {
    size_t len = 0;
    while (s[len]) len++;
    for (size_t i = 0; i < len / 2; i++) {
        char t = s[i];
        s[i] = s[len - 1 - i];
        s[len - 1 - i] = t;
    }
}""",
)

_s(
    "Write a C function that converts a null-terminated string to uppercase in-place.",
    """void str_toupper(char *s) {
    for (; *s; s++) {
        if (*s >= 'a' && *s <= 'z') *s -= 32;
    }
}""",
)

_s(
    "Write a C function that converts a null-terminated string to lowercase in-place.",
    """void str_tolower(char *s) {
    for (; *s; s++) {
        if (*s >= 'A' && *s <= 'Z') *s += 32;
    }
}""",
)

_s(
    "Write a C function that checks if a null-terminated string is a palindrome.",
    """#include <stddef.h>
#include <stdbool.h>
bool is_palindrome(const char *s) {
    size_t len = 0;
    while (s[len]) len++;
    for (size_t i = 0; i < len / 2; i++) {
        if (s[i] != s[len - 1 - i]) return false;
    }
    return true;
}""",
)

_s(
    "Write a C function that counts the number of words in a string, where words are separated by whitespace characters.",
    """#include <stdbool.h>
int word_count(const char *s) {
    int count = 0;
    bool in_word = false;
    for (; *s; s++) {
        if (*s == ' ' || *s == '\\t' || *s == '\\n') {
            in_word = false;
        } else if (!in_word) {
            in_word = true;
            count++;
        }
    }
    return count;
}""",
)

_s(
    "Write a C function that removes all occurrences of a given character from a string in-place.",
    """void remove_char(char *s, char c) {
    char *w = s;
    for (; *s; s++) {
        if (*s != c) *w++ = *s;
    }
    *w = '\\0';
}""",
)

_s(
    "Write a C function that trims leading whitespace from a string by returning a pointer into it.",
    """char *trim_left(char *s) {
    while (*s == ' ' || *s == '\\t' || *s == '\\n') s++;
    return s;
}""",
)

_s(
    "Write a C function that trims trailing whitespace from a string in-place by inserting null bytes.",
    """#include <stddef.h>
void trim_right(char *s) {
    size_t len = 0;
    while (s[len]) len++;
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\\t' || s[len - 1] == '\\n')) len--;
    s[len] = '\\0';
}""",
)

_s(
    "Write a C function that replaces all occurrences of character old_char with new_char in a string.",
    """void replace_char(char *s, char old_char, char new_char) {
    for (; *s; s++) {
        if (*s == old_char) *s = new_char;
    }
}""",
)

_s(
    "Write a C function that compresses consecutive spaces in a string into a single space in-place.",
    """void compress_spaces(char *s) {
    char *w = s;
    int prev_space = 0;
    for (; *s; s++) {
        if (*s == ' ') {
            if (!prev_space) *w++ = ' ';
            prev_space = 1;
        } else {
            *w++ = *s;
            prev_space = 0;
        }
    }
    *w = '\\0';
}""",
)

_s(
    "Write a C function that duplicates a string, allocating memory for the copy (strdup).",
    """#include <stdlib.h>
#include <stddef.h>
char *my_strdup(const char *s) {
    size_t len = 0;
    while (s[len]) len++;
    char *copy = (char *)malloc(len + 1);
    if (copy) {
        for (size_t i = 0; i <= len; i++) copy[i] = s[i];
    }
    return copy;
}""",
)

_s(
    "Write a C function that checks if string a starts with string b.",
    """#include <stdbool.h>
bool starts_with(const char *a, const char *b) {
    while (*b) {
        if (*a != *b) return false;
        a++; b++;
    }
    return true;
}""",
)

_s(
    "Write a C function that checks if string a ends with string b.",
    """#include <stddef.h>
#include <stdbool.h>
bool ends_with(const char *a, const char *b) {
    size_t la = 0, lb = 0;
    while (a[la]) la++;
    while (b[lb]) lb++;
    if (lb > la) return false;
    for (size_t i = 0; i < lb; i++) {
        if (a[la - lb + i] != b[i]) return false;
    }
    return true;
}""",
)

_s(
    "Write a C function that computes the Levenshtein distance between two short strings using dynamic programming.",
    """#include <string.h>
int min3(int a, int b, int c) {
    if (a <= b && a <= c) return a;
    if (b <= c) return b;
    return c;
}
int levenshtein(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b);
    int d[100][100];
    for (int i = 0; i <= la; i++) d[i][0] = i;
    for (int j = 0; j <= lb; j++) d[0][j] = j;
    for (int i = 1; i <= la; i++)
        for (int j = 1; j <= lb; j++)
            d[i][j] = min3(d[i-1][j] + 1,
                           d[i][j-1] + 1,
                           d[i-1][j-1] + (a[i-1] != b[j-1]));
    return d[la][lb];
}""",
)

_s(
    "Write a C function that finds the longest palindromic substring in a given string, returning its start index and length via output pointers.",
    """#include <stddef.h>
void longest_pal_substr(const char *s, int *start_out, int *len_out) {
    size_t n = 0;
    while (s[n]) n++;
    *start_out = 0; *len_out = n > 0 ? 1 : 0;
    for (size_t i = 0; i < n; i++) {
        for (int odd = 0; odd <= 1; odd++) {
            int l = (int)i, r = (int)i + odd;
            while (l >= 0 && r < (int)n && s[l] == s[r]) { l--; r++; }
            int cur_len = r - l - 1;
            if (cur_len > *len_out) { *start_out = l + 1; *len_out = cur_len; }
        }
    }
}""",
)

# ── Linked lists (20) ──────────────────────────────────────────────

_s(
    "Write a C function that reverses a singly-linked list in-place and returns the new head.",
    """struct Node { int val; struct Node *next; };
struct Node *reverse_list(struct Node *head) {
    struct Node *prev = 0, *cur = head, *next;
    while (cur) {
        next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
    }
    return prev;
}""",
)

_s(
    "Write a C function that returns the middle node of a singly-linked list using the fast-slow pointer technique.",
    """struct Node { int val; struct Node *next; };
struct Node *middle_node(struct Node *head) {
    struct Node *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}""",
)

_s(
    "Write a C function that detects if a singly-linked list has a cycle using Floyd's tortoise and hare algorithm.",
    """#include <stdbool.h>
struct Node { int val; struct Node *next; };
bool has_cycle(struct Node *head) {
    struct Node *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}""",
)

_s(
    "Write a C function that merges two sorted singly-linked lists into one sorted list.",
    """struct Node { int val; struct Node *next; };
struct Node *merge_sorted(struct Node *a, struct Node *b) {
    if (!a) return b; if (!b) return a;
    if (a->val <= b->val) { a->next = merge_sorted(a->next, b); return a; }
    else { b->next = merge_sorted(a, b->next); return b; }
}""",
)

_s(
    "Write a C function that merges two sorted singly-linked lists iteratively using a dummy head.",
    """struct Node { int val; struct Node *next; };
struct Node *merge_sorted_iter(struct Node *a, struct Node *b) {
    struct Node dummy = {0, 0}, *tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) { tail->next = a; a = a->next; }
        else { tail->next = b; b = b->next; }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}""",
)

_s(
    "Write a C function that removes the n-th node from the end of a singly-linked list and returns the head.",
    """struct Node { int val; struct Node *next; };
struct Node *remove_nth_from_end(struct Node *head, int n) {
    struct Node dummy = {0, head};
    struct Node *fast = &dummy, *slow = &dummy;
    for (int i = 0; i <= n; i++) fast = fast->next;
    while (fast) { slow = slow->next; fast = fast->next; }
    slow->next = slow->next->next;
    return dummy.next;
}""",
)

_s(
    "Write a C function that returns the node at the start of a cycle in a singly-linked list, if one exists.",
    """struct Node { int val; struct Node *next; };
struct Node *cycle_start(struct Node *head) {
    struct Node *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next; fast = fast->next->next;
        if (slow == fast) { slow = head;
            while (slow != fast) { slow = slow->next; fast = fast->next; }
            return slow;
        }
    }
    return 0;
}""",
)

_s(
    "Write a C function that removes all duplicate values from a sorted singly-linked list, keeping only distinct values.",
    """struct Node { int val; struct Node *next; };
struct Node *delete_duplicates_sorted(struct Node *head) {
    struct Node dummy = {0, head}, *prev = &dummy;
    while (head) {
        if (head->next && head->val == head->next->val) {
            while (head->next && head->val == head->next->val) head = head->next;
            prev->next = head->next;
        } else {
            prev = prev->next;
        }
        head = head->next;
    }
    return dummy.next;
}""",
)

_s(
    "Write a C function that splits a singly-linked list into two halves. Returns the head of the second half.",
    """struct Node { int val; struct Node *next; };
struct Node *split_list(struct Node *head) {
    struct Node *slow = head, *fast = head, *prev = 0;
    while (fast && fast->next) {
        prev = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    if (prev) prev->next = 0;
    return slow;
}""",
)

_s(
    "Write a C function that sorts a singly-linked list using merge sort.",
    """struct Node { int val; struct Node *next; };
struct Node *merge_sorted(struct Node *a, struct Node *b) {
    if (!a) return b; if (!b) return a;
    if (a->val <= b->val) { a->next = merge_sorted(a->next, b); return a; }
    else { b->next = merge_sorted(a, b->next); return b; }
}
struct Node *split_list(struct Node *head) {
    struct Node *slow = head, *fast = head, *prev = 0;
    while (fast && fast->next) { prev = slow; slow = slow->next; fast = fast->next->next; }
    if (prev) prev->next = 0;
    return slow;
}
struct Node *merge_sort_list(struct Node *head) {
    if (!head || !head->next) return head;
    struct Node *mid = split_list(head);
    struct Node *left = merge_sort_list(head);
    struct Node *right = merge_sort_list(mid);
    return merge_sorted(left, right);
}""",
)

_s(
    "Write a C function that finds the intersection point of two singly-linked lists (the node where they merge), if any.",
    """#include <stddef.h>
struct Node { int val; struct Node *next; };
struct Node *intersection(struct Node *a, struct Node *b) {
    struct Node *pa = a, *pb = b;
    while (pa != pb) {
        pa = pa ? pa->next : b;
        pb = pb ? pb->next : a;
    }
    return pa;
}""",
)

_s(
    "Write a C function that computes the length of a singly-linked list.",
    """struct Node { int val; struct Node *next; };
int list_len(struct Node *head) {
    int len = 0;
    while (head) { len++; head = head->next; }
    return len;
}""",
)

_s(
    "Write a C function that inserts a new node with a given value at the head of a singly-linked list and returns the new head.",
    """#include <stdlib.h>
struct Node { int val; struct Node *next; };
struct Node *prepend(struct Node *head, int val) {
    struct Node *n = (struct Node *)malloc(sizeof(struct Node));
    n->val = val; n->next = head;
    return n;
}""",
)

_s(
    "Write a C function that appends a node with a given value to the end of a singly-linked list and returns the head.",
    """#include <stdlib.h>
struct Node { int val; struct Node *next; };
struct Node *append(struct Node *head, int val) {
    struct Node *n = (struct Node *)malloc(sizeof(struct Node));
    n->val = val; n->next = 0;
    if (!head) return n;
    struct Node *cur = head;
    while (cur->next) cur = cur->next;
    cur->next = n;
    return head;
}""",
)

_s(
    "Write a C function that deletes the first occurrence of a given value from a singly-linked list and returns the head.",
    """#include <stdlib.h>
struct Node { int val; struct Node *next; };
struct Node *delete_val(struct Node *head, int val) {
    if (!head) return 0;
    if (head->val == val) { struct Node *rest = head->next; free(head); return rest; }
    struct Node *cur = head;
    while (cur->next && cur->next->val != val) cur = cur->next;
    if (cur->next) { struct Node *del = cur->next; cur->next = del->next; free(del); }
    return head;
}""",
)

_s(
    "Write a C function that checks if a singly-linked list is a palindrome (values form a palindrome).",
    """#include <stdbool.h>
struct Node { int val; struct Node *next; };
struct Node *reverse_list(struct Node *head) {
    struct Node *prev = 0, *cur = head, *next;
    while (cur) { next = cur->next; cur->next = prev; prev = cur; cur = next; }
    return prev;
}
bool is_list_palindrome(struct Node *head) {
    if (!head || !head->next) return true;
    struct Node *slow = head, *fast = head, *prev = 0;
    while (fast && fast->next) { prev = slow; slow = slow->next; fast = fast->next->next; }
    prev->next = 0;
    struct Node *second = reverse_list(slow);
    struct Node *a = head, *b = second;
    bool ok = true;
    while (a && b) { if (a->val != b->val) { ok = false; break; } a = a->next; b = b->next; }
    reverse_list(second);
    prev->next = slow;
    return ok;
}""",
)

_s(
    "Write a C function that removes every k-th node from a singly-linked list, where k >= 1. k=1 removes all nodes.",
    """struct Node { int val; struct Node *next; };
struct Node *remove_every_kth(struct Node *head, int k) {
    if (k <= 0 || !head) return head;
    if (k == 1) return 0;
    struct Node *cur = head;
    int pos = 1;
    while (cur && cur->next) {
        if (pos == k - 1) { cur->next = cur->next->next; pos = 1; }
        else { pos++; }
        cur = cur->next;
    }
    return head;
}""",
)

_s(
    "Write a C function that swaps nodes pairwise in a singly-linked list (1<->2, 3<->4, ...).",
    """struct Node { int val; struct Node *next; };
struct Node *swap_pairs(struct Node *head) {
    if (!head || !head->next) return head;
    struct Node *new_head = head->next;
    head->next = swap_pairs(new_head->next);
    new_head->next = head;
    return new_head;
}""",
)

_s(
    "Write a C function that groups all odd-indexed nodes followed by even-indexed nodes in a singly-linked list (1-indexed).",
    """struct Node { int val; struct Node *next; };
struct Node *odd_even_list(struct Node *head) {
    if (!head) return 0;
    struct Node *odd = head, *even = head->next, *even_h = even;
    while (even && even->next) {
        odd->next = even->next;
        odd = odd->next;
        even->next = odd->next;
        even = even->next;
    }
    odd->next = even_h;
    return head;
}""",
)

_s(
    "Write a C function that removes duplicates from an unsorted singly-linked list using a nested loop.",
    """#include <stdlib.h>
struct Node { int val; struct Node *next; };
struct Node *remove_dups_unsorted(struct Node *head) {
    struct Node *cur = head;
    while (cur) {
        struct Node *runner = cur;
        while (runner->next) {
            if (runner->next->val == cur->val) {
                struct Node *dup = runner->next;
                runner->next = dup->next;
                free(dup);
            } else {
                runner = runner->next;
            }
        }
        cur = cur->next;
    }
    return head;
}""",
)

# ── Arrays and searching (25) ──────────────────────────────────────

_s(
    "Write a C function that performs binary search on a sorted array of integers, returning the index or -1 if not found.",
    """int binary_search(const int *arr, int len, int target) {
    int lo = 0, hi = len - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}""",
)

_s(
    "Write a C function that finds the maximum value in an integer array.",
    """int array_max(const int *arr, int len) {
    int max = arr[0];
    for (int i = 1; i < len; i++)
        if (arr[i] > max) max = arr[i];
    return max;
}""",
)

_s(
    "Write a C function that finds the minimum value in an integer array.",
    """int array_min(const int *arr, int len) {
    int min = arr[0];
    for (int i = 1; i < len; i++)
        if (arr[i] < min) min = arr[i];
    return min;
}""",
)

_s(
    "Write a C function that computes the sum of all elements in an integer array.",
    """long long array_sum(const int *arr, int len) {
    long long total = 0;
    for (int i = 0; i < len; i++) total += arr[i];
    return total;
}""",
)

_s(
    "Write a C function that computes the average of an integer array as a double.",
    """double array_avg(const int *arr, int len) {
    if (len == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < len; i++) sum += (double)arr[i];
    return sum / len;
}""",
)

_s(
    "Write a C function that reverses an array of integers in-place.",
    """void reverse_array(int *arr, int len) {
    for (int i = 0; i < len / 2; i++) {
        int t = arr[i];
        arr[i] = arr[len - 1 - i];
        arr[len - 1 - i] = t;
    }
}""",
)

_s(
    "Write a C function that rotates an integer array left by k positions in-place.",
    """void reverse_range(int *a, int lo, int hi) {
    while (lo < hi) { int t = a[lo]; a[lo] = a[hi]; a[hi] = t; lo++; hi--; }
}
void rotate_left(int *arr, int len, int k) {
    k %= len;
    if (k == 0) return;
    reverse_range(arr, 0, k - 1);
    reverse_range(arr, k, len - 1);
    reverse_range(arr, 0, len - 1);
}""",
)

_s(
    "Write a C function that finds the first occurrence of a target integer in an unsorted array, returning the index or -1.",
    """int linear_search(const int *arr, int len, int target) {
    for (int i = 0; i < len; i++)
        if (arr[i] == target) return i;
    return -1;
}""",
)

_s(
    "Write a C function that finds the last occurrence of a target integer in an array, scanning from the end.",
    """int linear_search_last(const int *arr, int len, int target) {
    for (int i = len - 1; i >= 0; i--)
        if (arr[i] == target) return i;
    return -1;
}""",
)

_s(
    "Write a C function that merges two sorted integer arrays into a third sorted array.",
    """void merge_sorted_arrays(const int *a, int la, const int *b, int lb, int *out) {
    int i = 0, j = 0, k = 0;
    while (i < la && j < lb) {
        out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    }
    while (i < la) out[k++] = a[i++];
    while (j < lb) out[k++] = b[j++];
}""",
)

_s(
    "Write a C function that implements insertion sort on an integer array in-place.",
    """void insertion_sort(int *arr, int len) {
    for (int i = 1; i < len; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j + 1] = arr[j]; j--; }
        arr[j + 1] = key;
    }
}""",
)

_s(
    "Write a C function that implements selection sort on an integer array in-place.",
    """void selection_sort(int *arr, int len) {
    for (int i = 0; i < len - 1; i++) {
        int min_i = i;
        for (int j = i + 1; j < len; j++)
            if (arr[j] < arr[min_i]) min_i = j;
        if (min_i != i) { int t = arr[i]; arr[i] = arr[min_i]; arr[min_i] = t; }
    }
}""",
)

_s(
    "Write a C function that implements bubble sort on an integer array in-place with early termination.",
    """#include <stdbool.h>
void bubble_sort(int *arr, int len) {
    for (int i = 0; i < len - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < len - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                int t = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = t;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}""",
)

_s(
    "Write a C function that implements the partition step of quicksort (Lomuto scheme). Returns the pivot index.",
    """int partition(int *arr, int lo, int hi) {
    int pivot = arr[hi], i = lo - 1;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) { i++; int t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    }
    int t = arr[i + 1]; arr[i + 1] = arr[hi]; arr[hi] = t;
    return i + 1;
}""",
)

_s(
    "Write a C function that computes the maximum subarray sum using Kadane's algorithm.",
    """int max_subarray_sum(const int *arr, int len) {
    int max_here = arr[0], max_sofar = arr[0];
    for (int i = 1; i < len; i++) {
        max_here = (max_here + arr[i] > arr[i]) ? max_here + arr[i] : arr[i];
        if (max_here > max_sofar) max_sofar = max_here;
    }
    return max_sofar;
}""",
)

_s(
    "Write a C function that finds two indices whose values sum to a target in a sorted array. Returns 1 if found, 0 otherwise. Writes indices to output pointers.",
    """#include <stdbool.h>
int two_sum_sorted(const int *arr, int len, int target, int *i_out, int *j_out) {
    int i = 0, j = len - 1;
    while (i < j) {
        int sum = arr[i] + arr[j];
        if (sum == target) { *i_out = i; *j_out = j; return 1; }
        if (sum < target) i++; else j--;
    }
    return 0;
}""",
)

_s(
    "Write a C function that removes all occurrences of a value from an integer array in-place. Returns new length.",
    """int remove_value(int *arr, int len, int val) {
    int w = 0;
    for (int i = 0; i < len; i++)
        if (arr[i] != val) arr[w++] = arr[i];
    return w;
}""",
)

_s(
    "Write a C function that removes duplicates from a sorted integer array in-place. Returns new length.",
    """int remove_duplicates_sorted(int *arr, int len) {
    if (len <= 1) return len;
    int w = 1;
    for (int i = 1; i < len; i++)
        if (arr[i] != arr[w - 1]) arr[w++] = arr[i];
    return w;
}""",
)

_s(
    "Write a C function that moves all zeros in an integer array to the end while preserving the relative order of non-zero elements.",
    """void move_zeros_to_end(int *arr, int len) {
    int w = 0;
    for (int i = 0; i < len; i++)
        if (arr[i] != 0) arr[w++] = arr[i];
    while (w < len) arr[w++] = 0;
}""",
)

_s(
    "Write a C function that finds the floor (greatest element <= target) in a sorted integer array using binary search.",
    """int floor_sorted(const int *arr, int len, int target) {
    int lo = 0, hi = len - 1, result = -1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] <= target) { result = arr[mid]; lo = mid + 1; }
        else hi = mid - 1;
    }
    return result;
}""",
)

_s(
    "Write a C function that finds the ceiling (smallest element >= target) in a sorted integer array using binary search.",
    """int ceil_sorted(const int *arr, int len, int target) {
    int lo = 0, hi = len - 1, result = -1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] >= target) { result = arr[mid]; hi = mid - 1; }
        else lo = mid + 1;
    }
    return result;
}""",
)

_s(
    "Write a C function that finds the majority element in an array (appears > n/2 times) using Boyer-Moore majority vote. Assumes one exists.",
    """int majority_element(const int *arr, int len) {
    int cand = arr[0], count = 1;
    for (int i = 1; i < len; i++) {
        if (count == 0) { cand = arr[i]; count = 1; }
        else if (arr[i] == cand) count++;
        else count--;
    }
    return cand;
}""",
)

_s(
    "Write a C function that finds a peak element (an element greater than its neighbors) in an unsorted integer array using binary search. Returns any peak index.",
    """int find_peak(const int *arr, int len) {
    int lo = 0, hi = len - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < arr[mid + 1]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}""",
)

_s(
    "Write a C function that computes the product of all elements except self for each position in an integer array. Stores result in a pre-allocated output array.",
    """void product_except_self(const int *arr, int len, int *out) {
    out[0] = 1;
    for (int i = 1; i < len; i++) out[i] = out[i - 1] * arr[i - 1];
    int right = 1;
    for (int i = len - 1; i >= 0; i--) { out[i] *= right; right *= arr[i]; }
}""",
)

# ── Stacks and queues (10) ─────────────────────────────────────────

_s(
    "Write a C function implementing a stack using an array with push, pop, and peek operations. The stack is represented as a struct with a fixed-size buffer, top index, and capacity.",
    """#include <stdbool.h>
#define MAX_STACK 256
struct Stack { int buf[MAX_STACK]; int top; int cap; };
void stack_init(struct Stack *s, int cap) { s->top = -1; s->cap = cap; }
bool stack_push(struct Stack *s, int val) {
    if (s->top + 1 >= s->cap) return false;
    s->buf[++s->top] = val; return true;
}
bool stack_pop(struct Stack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->buf[s->top--]; return true;
}
bool stack_peek(const struct Stack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->buf[s->top]; return true;
}
bool stack_empty(const struct Stack *s) { return s->top < 0; }""",
)

_s(
    "Write a C function implementing a circular queue using an array with enqueue and dequeue. The queue struct holds buffer, head, tail, size, and capacity.",
    """#include <stdbool.h>
#define MAX_QUEUE 256
struct Queue { int buf[MAX_QUEUE]; int head; int tail; int sz; int cap; };
void q_init(struct Queue *q, int cap) { q->head = 0; q->tail = 0; q->sz = 0; q->cap = cap; }
bool q_enq(struct Queue *q, int val) {
    if (q->sz >= q->cap) return false;
    q->buf[q->tail] = val;
    q->tail = (q->tail + 1) % q->cap;
    q->sz++; return true;
}
bool q_deq(struct Queue *q, int *out) {
    if (q->sz == 0) return false;
    *out = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    q->sz--; return true;
}
bool q_empty(const struct Queue *q) { return q->sz == 0; }""",
)

_s(
    "Write a C function that checks if a string containing parentheses '()', brackets '[]', and braces '{}' is balanced.",
    """#include <stdbool.h>
#define MAX_DEPTH 256
bool is_balanced(const char *s) {
    char stack[MAX_DEPTH]; int top = -1;
    for (; *s; s++) {
        if (*s == '(' || *s == '[' || *s == '{') { stack[++top] = *s; continue; }
        if (*s == ')' || *s == ']' || *s == '}') {
            if (top < 0) return false;
            char open = stack[top--];
            if ((*s == ')' && open != '(') || (*s == ']' && open != '[') || (*s == '}' && open != '{'))
                return false;
        }
    }
    return top < 0;
}""",
)

_s(
    "Write a C function that evaluates a postfix (Reverse Polish Notation) expression containing integers and operators +, -, *. Returns the result.",
    """#include <ctype.h>
#define MAX_RPN 128
int eval_rpn(const char *s) {
    int stack[MAX_RPN]; int top = -1;
    while (*s) {
        if (*s == ' ') { s++; continue; }
        if (isdigit(*s) || (*s == '-' && isdigit(*(s+1)))) {
            int sign = 1;
            if (*s == '-') { sign = -1; s++; }
            int num = 0;
            while (isdigit(*s)) { num = num * 10 + (*s - '0'); s++; }
            stack[++top] = sign * num;
        } else {
            int b = stack[top--], a = stack[top--];
            if (*s == '+') stack[++top] = a + b;
            else if (*s == '-') stack[++top] = a - b;
            else if (*s == '*') stack[++top] = a * b;
            s++;
        }
    }
    return stack[0];
}""",
)

_s(
    "Write a C function that simulates a min-stack supporting push, pop, top, and getMin in O(1) each.",
    """#include <stdbool.h>
#define MAX_MINSTACK 256
struct MinStack { int val[MAX_MINSTACK]; int min[MAX_MINSTACK]; int top; int cap; };
void ms_init(struct MinStack *s, int cap) { s->top = -1; s->cap = cap; }
bool ms_push(struct MinStack *s, int v) {
    if (s->top + 1 >= s->cap) return false;
    int new_min = (s->top >= 0 && s->min[s->top] < v) ? s->min[s->top] : v;
    s->top++;
    s->val[s->top] = v; s->min[s->top] = new_min;
    return true;
}
bool ms_pop(struct MinStack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->val[s->top--]; return true;
}
bool ms_top(const struct MinStack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->val[s->top]; return true;
}
bool ms_get_min(const struct MinStack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->min[s->top]; return true;
}""",
)

_s(
    "Write a C function that implements a stack which supports push, pop, and finding the maximum element in O(1).",
    """#include <stdbool.h>
#define MAX_MAXSTACK 256
struct MaxStack { int val[MAX_MAXSTACK]; int max[MAX_MAXSTACK]; int top; int cap; };
void mxs_init(struct MaxStack *s, int cap) { s->top = -1; s->cap = cap; }
bool mxs_push(struct MaxStack *s, int v) {
    if (s->top + 1 >= s->cap) return false;
    int new_max = (s->top >= 0 && s->max[s->top] > v) ? s->max[s->top] : v;
    s->top++;
    s->val[s->top] = v; s->max[s->top] = new_max;
    return true;
}
bool mxs_pop(struct MaxStack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->val[s->top--]; return true;
}
bool mxs_get_max(const struct MaxStack *s, int *out) {
    if (s->top < 0) return false;
    *out = s->max[s->top]; return true;
}""",
)

_s(
    "Write a C function that implements a basic priority queue (max-heap) using an array with push and pop operations.",
    """#include <stdbool.h>
#define MAX_HEAP 256
struct PrioQ { int heap[MAX_HEAP]; int sz; int cap; };
void pq_init(struct PrioQ *q, int cap) { q->sz = 0; q->cap = cap; }
static void pq_swim(struct PrioQ *q, int i) {
    while (i > 0) { int p = (i - 1) / 2; if (q->heap[p] >= q->heap[i]) break;
        int t = q->heap[p]; q->heap[p] = q->heap[i]; q->heap[i] = t; i = p; }
}
bool pq_push(struct PrioQ *q, int v) {
    if (q->sz >= q->cap) return false;
    q->heap[q->sz] = v; pq_swim(q, q->sz); q->sz++; return true;
}
static void pq_sink(struct PrioQ *q, int i) {
    while (1) { int big = i, l = 2 * i + 1, r = 2 * i + 2;
        if (l < q->sz && q->heap[l] > q->heap[big]) big = l;
        if (r < q->sz && q->heap[r] > q->heap[big]) big = r;
        if (big == i) break;
        int t = q->heap[i]; q->heap[i] = q->heap[big]; q->heap[big] = t; i = big; }
}
bool pq_pop(struct PrioQ *q, int *out) {
    if (q->sz == 0) return false;
    *out = q->heap[0]; q->heap[0] = q->heap[--q->sz]; pq_sink(q, 0); return true;
}""",
)

_s(
    "Write a C function that evaluates an arithmetic expression containing +, -, *, / and parentheses, returning the integer result following operator precedence.",
    """#include <ctype.h>
int calc_expr(const char *s, int *pos);
int calc_term(const char *s, int *pos);
int calc_num(const char *s, int *pos) {
    int n = 0;
    while (isdigit(s[*pos])) { n = n * 10 + (s[*pos] - '0'); (*pos)++; }
    return n;
}
int calc_factor(const char *s, int *pos) {
    if (s[*pos] == '(') { (*pos)++; int r = calc_expr(s, pos); (*pos)++; return r; }
    return calc_num(s, pos);
}
int calc_term(const char *s, int *pos) {
    int r = calc_factor(s, pos);
    while (s[*pos] == '*' || s[*pos] == '/') {
        char op = s[*pos]; (*pos)++;
        int rhs = calc_factor(s, pos);
        r = (op == '*') ? r * rhs : r / rhs;
    }
    return r;
}
int calc_expr(const char *s, int *pos) {
    int r = calc_term(s, pos);
    while (s[*pos] == '+' || s[*pos] == '-') {
        char op = s[*pos]; (*pos)++;
        int rhs = calc_term(s, pos);
        r = (op == '+') ? r + rhs : r - rhs;
    }
    return r;
}
int calculate(const char *s) { int pos = 0; return calc_expr(s, &pos); }""",
)

_s(
    "Write a C function that computes the next greater element for each position in an integer array using a monotonic decreasing stack. Stores results in output array, or -1 if none.",
    """void next_greater(const int *arr, int len, int *out) {
    int stack[256], top = -1;
    for (int i = len - 1; i >= 0; i--) {
        while (top >= 0 && stack[top] <= arr[i]) top--;
        out[i] = (top >= 0) ? stack[top] : -1;
        stack[++top] = arr[i];
    }
}""",
)

_s(
    "Write a C function that implements a deque (double-ended queue) using a circular array with push/pop from both ends.",
    """#include <stdbool.h>
#define MAX_DEQUE 256
struct Deque { int buf[MAX_DEQUE]; int head; int tail; int sz; int cap; };
void dq_init(struct Deque *q, int cap) { q->head = 0; q->tail = 0; q->sz = 0; q->cap = cap; }
bool dq_push_front(struct Deque *q, int v) {
    if (q->sz >= q->cap) return false;
    q->head = (q->head - 1 + q->cap) % q->cap;
    q->buf[q->head] = v; q->sz++; return true;
}
bool dq_push_back(struct Deque *q, int v) {
    if (q->sz >= q->cap) return false;
    q->buf[q->tail] = v;
    q->tail = (q->tail + 1) % q->cap; q->sz++; return true;
}
bool dq_pop_front(struct Deque *q, int *out) {
    if (q->sz == 0) return false;
    *out = q->buf[q->head]; q->head = (q->head + 1) % q->cap; q->sz--; return true;
}
bool dq_pop_back(struct Deque *q, int *out) {
    if (q->sz == 0) return false;
    q->tail = (q->tail - 1 + q->cap) % q->cap;
    *out = q->buf[q->tail]; q->sz--; return true;
}""",
)

# ── Trees (20) ─────────────────────────────────────────────────────

_s(
    "Write a C function that performs an in-order traversal of a binary tree, calling a callback function for each node's value.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
typedef void (*Visitor)(int);
void inorder(struct TreeNode *root, Visitor visit) {
    if (!root) return;
    inorder(root->left, visit);
    visit(root->val);
    inorder(root->right, visit);
}""",
)

_s(
    "Write a C function that performs a pre-order traversal of a binary tree, calling a callback function for each node's value.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
typedef void (*Visitor)(int);
void preorder(struct TreeNode *root, Visitor visit) {
    if (!root) return;
    visit(root->val);
    preorder(root->left, visit);
    preorder(root->right, visit);
}""",
)

_s(
    "Write a C function that performs a post-order traversal of a binary tree, calling a callback function for each node's value.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
typedef void (*Visitor)(int);
void postorder(struct TreeNode *root, Visitor visit) {
    if (!root) return;
    postorder(root->left, visit);
    postorder(root->right, visit);
    visit(root->val);
}""",
)

_s(
    "Write a C function that computes the maximum depth of a binary tree.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
int max_depth(struct TreeNode *root) {
    if (!root) return 0;
    int ld = max_depth(root->left);
    int rd = max_depth(root->right);
    return 1 + (ld > rd ? ld : rd);
}""",
)

_s(
    "Write a C function that checks if a binary tree is balanced (heights of left and right subtrees differ by at most 1 at every node).",
    """#include <stdbool.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
int check_bal(struct TreeNode *root, bool *ok) {
    if (!root) return 0;
    int lh = check_bal(root->left, ok);
    int rh = check_bal(root->right, ok);
    if (lh - rh > 1 || rh - lh > 1) *ok = false;
    return 1 + (lh > rh ? lh : rh);
}
bool is_balanced(struct TreeNode *root) { bool ok = true; check_bal(root, &ok); return ok; }""",
)

_s(
    "Write a C function that inserts a value into a binary search tree (BST). Returns the new root.",
    """#include <stdlib.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *bst_insert(struct TreeNode *root, int val) {
    if (!root) {
        struct TreeNode *n = (struct TreeNode *)malloc(sizeof(struct TreeNode));
        n->val = val; n->left = 0; n->right = 0; return n;
    }
    if (val < root->val) root->left = bst_insert(root->left, val);
    else root->right = bst_insert(root->right, val);
    return root;
}""",
)

_s(
    "Write a C function that searches for a value in a BST. Returns a pointer to the node or NULL.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *bst_search(struct TreeNode *root, int val) {
    while (root) {
        if (val == root->val) return root;
        root = (val < root->val) ? root->left : root->right;
    }
    return 0;
}""",
)

_s(
    "Write a C function that finds the minimum value in a BST.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *bst_min(struct TreeNode *root) {
    while (root && root->left) root = root->left;
    return root;
}""",
)

_s(
    "Write a C function that finds the maximum value in a BST.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *bst_max(struct TreeNode *root) {
    while (root && root->right) root = root->left;
    return root;
}""",
)

_s(
    "Write a C function that finds the lowest common ancestor of two nodes in a BST.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *bst_lca(struct TreeNode *root, int p, int q) {
    if (!root) return 0;
    if (p < root->val && q < root->val) return bst_lca(root->left, p, q);
    if (p > root->val && q > root->val) return bst_lca(root->right, p, q);
    return root;
}""",
)

_s(
    "Write a C function that counts the number of nodes in a binary tree.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
int tree_size(struct TreeNode *root) {
    if (!root) return 0;
    return 1 + tree_size(root->left) + tree_size(root->right);
}""",
)

_s(
    "Write a C function that checks if a binary tree is a mirror of itself (symmetric).",
    """#include <stdbool.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
bool sym(struct TreeNode *a, struct TreeNode *b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && sym(a->left, b->right) && sym(a->right, b->left);
}
bool is_symmetric(struct TreeNode *root) { return sym(root, root); }""",
)

_s(
    "Write a C function that checks if two binary trees are identical in structure and values.",
    """#include <stdbool.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
bool same_tree(struct TreeNode *a, struct TreeNode *b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && same_tree(a->left, b->left) && same_tree(a->right, b->right);
}""",
)

_s(
    "Write a C function that converts a sorted array to a height-balanced BST. Returns the root.",
    """#include <stdlib.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *build_bst(const int *arr, int lo, int hi) {
    if (lo > hi) return 0;
    int mid = lo + (hi - lo) / 2;
    struct TreeNode *n = (struct TreeNode *)malloc(sizeof(struct TreeNode));
    n->val = arr[mid];
    n->left = build_bst(arr, lo, mid - 1);
    n->right = build_bst(arr, mid + 1, hi);
    return n;
}""",
)

_s(
    "Write a C function that returns all values at a given depth in a binary tree (level order). Stores them in a pre-allocated output array and returns the count.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
#define MAX_NODES 512
int level_order(struct TreeNode *root, int *out) {
    if (!root) return 0;
    struct TreeNode *q[MAX_NODES]; int head = 0, tail = 0, count = 0;
    q[tail++] = root;
    while (head < tail) {
        struct TreeNode *n = q[head++];
        out[count++] = n->val;
        if (n->left) q[tail++] = n->left;
        if (n->right) q[tail++] = n->right;
    }
    return count;
}""",
)

_s(
    "Write a C function that computes the sum of all leaf node values in a binary tree.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
int leaf_sum(struct TreeNode *root) {
    if (!root) return 0;
    if (!root->left && !root->right) return root->val;
    return leaf_sum(root->left) + leaf_sum(root->right);
}""",
)

_s(
    "Write a C function that checks if a binary tree is a valid BST within the range (min_val, max_val).",
    """#include <stdbool.h>
#include <limits.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
bool valid_bst_range(struct TreeNode *root, long lo, long hi) {
    if (!root) return true;
    if (root->val <= lo || root->val >= hi) return false;
    return valid_bst_range(root->left, lo, root->val) &&
           valid_bst_range(root->right, root->val, hi);
}
bool is_valid_bst(struct TreeNode *root) {
    return valid_bst_range(root, (long)INT_MIN - 1, (long)INT_MAX + 1);
}""",
)

_s(
    "Write a C function that deletes a node with a given value from a BST. Returns the new root.",
    """#include <stdlib.h>
struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *bst_min(struct TreeNode *root) {
    while (root && root->left) root = root->left; return root;
}
struct TreeNode *bst_delete(struct TreeNode *root, int val) {
    if (!root) return 0;
    if (val < root->val) root->left = bst_delete(root->left, val);
    else if (val > root->val) root->right = bst_delete(root->right, val);
    else {
        if (!root->left) { struct TreeNode *r = root->right; free(root); return r; }
        if (!root->right) { struct TreeNode *l = root->left; free(root); return l; }
        struct TreeNode *succ = bst_min(root->right);
        root->val = succ->val;
        root->right = bst_delete(root->right, succ->val);
    }
    return root;
}""",
)

_s(
    "Write a C function that finds the diameter of a binary tree (longest path between any two nodes), measured in edges.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
int diam_dfs(struct TreeNode *root, int *max_d) {
    if (!root) return 0;
    int lh = diam_dfs(root->left, max_d);
    int rh = diam_dfs(root->right, max_d);
    if (lh + rh > *max_d) *max_d = lh + rh;
    return 1 + (lh > rh ? lh : rh);
}
int tree_diameter(struct TreeNode *root) { int md = 0; diam_dfs(root, &md); return md; }""",
)

_s(
    "Write a C function that inverts a binary tree (mirrors it) in-place. Returns the root.",
    """struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
struct TreeNode *invert_tree(struct TreeNode *root) {
    if (!root) return 0;
    struct TreeNode *t = root->left;
    root->left = invert_tree(root->right);
    root->right = invert_tree(t);
    return root;
}""",
)

# ── Sorting (10) ───────────────────────────────────────────────────

_s(
    "Write a C function that implements merge sort on an integer array. Uses a temporary buffer for merging.",
    """void merge(int *arr, int lo, int mid, int hi, int *tmp) {
    int i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= mid) tmp[k++] = arr[i++];
    while (j <= hi)  tmp[k++] = arr[j++];
    for (i = lo; i <= hi; i++) arr[i] = tmp[i];
}
void msort(int *arr, int lo, int hi, int *tmp) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    msort(arr, lo, mid, tmp);
    msort(arr, mid + 1, hi, tmp);
    merge(arr, lo, mid, hi, tmp);
}
void merge_sort(int *arr, int len) {
    int tmp[1024];
    msort(arr, 0, len - 1, tmp);
}""",
)

_s(
    "Write a C function that implements quicksort on an integer array using Lomuto partition and recursion.",
    """int partition(int *arr, int lo, int hi) {
    int pivot = arr[hi], i = lo - 1;
    for (int j = lo; j < hi; j++)
        if (arr[j] <= pivot) { i++; int t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    int t = arr[i + 1]; arr[i + 1] = arr[hi]; arr[hi] = t;
    return i + 1;
}
void qs(int *arr, int lo, int hi) {
    if (lo >= hi) return;
    int p = partition(arr, lo, hi);
    qs(arr, lo, p - 1);
    qs(arr, p + 1, hi);
}
void quicksort(int *arr, int len) { qs(arr, 0, len - 1); }""",
)

_s(
    "Write a C function that implements counting sort for non-negative integers up to a known maximum value.",
    """#include <stdlib.h>
void counting_sort(int *arr, int len, int max_val) {
    int *count = (int *)calloc(max_val + 1, sizeof(int));
    for (int i = 0; i < len; i++) count[arr[i]]++;
    int w = 0;
    for (int v = 0; v <= max_val; v++)
        while (count[v]-- > 0) arr[w++] = v;
    free(count);
}""",
)

_s(
    "Write a C function that implements heapsort on an integer array in-place using a max-heap.",
    """void heapify(int *arr, int len, int i) {
    int largest = i, l = 2 * i + 1, r = 2 * i + 2;
    if (l < len && arr[l] > arr[largest]) largest = l;
    if (r < len && arr[r] > arr[largest]) largest = r;
    if (largest != i) { int t = arr[i]; arr[i] = arr[largest]; arr[largest] = t; heapify(arr, len, largest); }
}
void heapsort(int *arr, int len) {
    for (int i = len / 2 - 1; i >= 0; i--) heapify(arr, len, i);
    for (int i = len - 1; i > 0; i--) {
        int t = arr[0]; arr[0] = arr[i]; arr[i] = t;
        heapify(arr, i, 0);
    }
}""",
)

_s(
    "Write a C function that implements shell sort on an integer array using the Knuth gap sequence: (3^k - 1)/2.",
    """void shell_sort(int *arr, int len) {
    int gap = 1;
    while (gap < len / 3) gap = 3 * gap + 1;
    for (; gap > 0; gap /= 3) {
        for (int i = gap; i < len; i++) {
            int key = arr[i], j = i;
            while (j >= gap && arr[j - gap] > key) { arr[j] = arr[j - gap]; j -= gap; }
            arr[j] = key;
        }
    }
}""",
)

_s(
    "Write a C function that finds the k-th smallest element in an unsorted array using the quickselect algorithm (Hoare's selection).",
    """int partition(int *arr, int lo, int hi) {
    int pivot = arr[hi], i = lo - 1;
    for (int j = lo; j < hi; j++) if (arr[j] <= pivot) { i++; int t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    int t = arr[i + 1]; arr[i + 1] = arr[hi]; arr[hi] = t; return i + 1;
}
int quickselect(int *arr, int len, int k) {
    int lo = 0, hi = len - 1;
    while (lo <= hi) {
        int p = partition(arr, lo, hi);
        if (p == k) return arr[p];
        if (p < k) lo = p + 1; else hi = p - 1;
    }
    return -1;
}""",
)

_s(
    "Write a C function that implements radix sort for non-negative 32-bit integers using LSD (least significant digit) approach with base 256.",
    """#include <string.h>
#include <stdlib.h>
void radix_sort_u32(unsigned int *arr, int len) {
    unsigned int *tmp = (unsigned int *)malloc(len * sizeof(unsigned int));
    for (int shift = 0; shift < 32; shift += 8) {
        int count[256] = {0};
        for (int i = 0; i < len; i++) count[(arr[i] >> shift) & 0xFF]++;
        for (int i = 1; i < 256; i++) count[i] += count[i - 1];
        for (int i = len - 1; i >= 0; i--) tmp[--count[(arr[i] >> shift) & 0xFF]] = arr[i];
        for (int i = 0; i < len; i++) arr[i] = tmp[i];
    }
    free(tmp);
}""",
)

_s(
    "Write a C function that sorts three integers in-place in ascending order using comparison-based swap network.",
    """void sort3(int *a, int *b, int *c) {
    int t;
    if (*a > *b) { t = *a; *a = *b; *b = t; }
    if (*b > *c) { t = *b; *b = *c; *c = t; }
    if (*a > *b) { t = *a; *a = *b; *b = t; }
}""",
)

_s(
    "Write a C function that checks if an integer array is sorted in non-decreasing order.",
    """#include <stdbool.h>
bool is_sorted(const int *arr, int len) {
    for (int i = 1; i < len; i++)
        if (arr[i] < arr[i - 1]) return false;
    return true;
}""",
)

_s(
    "Write a C function that finds the minimum number of swaps needed to sort a binary array (containing only 0s and 1s).",
    """int min_swaps_binary(int *arr, int len) {
    int zeros = 0;
    for (int i = 0; i < len; i++) if (arr[i] == 0) zeros++;
    int bad = 0;
    for (int i = 0; i < zeros; i++) if (arr[i] == 1) bad++;
    return bad;
}""",
)

# ── Hash tables (10) ───────────────────────────────────────────────

_s(
    "Write a C function that computes a simple multiplicative hash of a string (djb2 algorithm).",
    """unsigned long djb2(const char *s) {
    unsigned long hash = 5381;
    int c;
    while ((c = *s++)) hash = ((hash << 5) + hash) + (unsigned char)c;
    return hash;
}""",
)

_s(
    "Write a C function that computes the FNV-1a hash of a byte buffer.",
    """#include <stdint.h>
#include <stddef.h>
uint32_t fnv1a(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint32_t hash = 0x811C9DC5u;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint32_t)p[i];
        hash *= 0x01000193u;
    }
    return hash;
}""",
)

_s(
    "Write a C function implementing a simple hash table with separate chaining for integer keys and values. Supports insert, lookup, and delete.",
    """#include <stdlib.h>
#include <stdbool.h>
#define HT_CAP 64
struct HTEntry { int key; int val; struct HTEntry *next; };
struct HashTable { struct HTEntry *buckets[HT_CAP]; };
unsigned int ht_hash(int key) { return (unsigned int)key % HT_CAP; }
void ht_init(struct HashTable *t) { for (int i = 0; i < HT_CAP; i++) t->buckets[i] = 0; }
void ht_put(struct HashTable *t, int key, int val) {
    unsigned int idx = ht_hash(key);
    struct HTEntry *e = t->buckets[idx];
    while (e) { if (e->key == key) { e->val = val; return; } e = e->next; }
    e = (struct HTEntry *)malloc(sizeof(struct HTEntry));
    e->key = key; e->val = val; e->next = t->buckets[idx]; t->buckets[idx] = e;
}
bool ht_get(struct HashTable *t, int key, int *out) {
    for (struct HTEntry *e = t->buckets[ht_hash(key)]; e; e = e->next)
        if (e->key == key) { *out = e->val; return true; }
    return false;
}
bool ht_del(struct HashTable *t, int key) {
    unsigned int idx = ht_hash(key);
    struct HTEntry *prev = 0, *e = t->buckets[idx];
    while (e) {
        if (e->key == key) {
            if (prev) prev->next = e->next; else t->buckets[idx] = e->next;
            free(e); return true;
        }
        prev = e; e = e->next;
    }
    return false;
}""",
)

_s(
    "Write a C function implementing hash table with linear probing (open addressing) for integer keys and values. Supports insert and lookup.",
    """#include <stdbool.h>
#define OP_CAP 128
struct Opslot { int key; int val; int used; };
struct OpenHT { struct Opslot slots[OP_CAP]; };
void oht_init(struct OpenHT *t) { for (int i = 0; i < OP_CAP; i++) t->slots[i].used = 0; }
bool oht_put(struct OpenHT *t, int key, int val) {
    unsigned int h = (unsigned int)key % OP_CAP;
    for (int i = 0; i < OP_CAP; i++) {
        int idx = (int)((h + (unsigned int)i) % OP_CAP);
        if (!t->slots[idx].used || t->slots[idx].key == key) {
            t->slots[idx].key = key; t->slots[idx].val = val; t->slots[idx].used = 1;
            return true;
        }
    }
    return false;
}
bool oht_get(struct OpenHT *t, int key, int *out) {
    unsigned int h = (unsigned int)key % OP_CAP;
    for (int i = 0; i < OP_CAP; i++) {
        int idx = (int)((h + (unsigned int)i) % OP_CAP);
        if (!t->slots[idx].used) return false;
        if (t->slots[idx].key == key) { *out = t->slots[idx].val; return true; }
    }
    return false;
}""",
)

_s(
    "Write a C function that finds the first non-repeating character in a string and returns its index, or -1 if none.",
    """int first_uniq_char(const char *s) {
    int count[256] = {0};
    for (int i = 0; s[i]; i++) count[(unsigned char)s[i]]++;
    for (int i = 0; s[i]; i++)
        if (count[(unsigned char)s[i]] == 1) return i;
    return -1;
}""",
)

_s(
    "Write a C function that checks if two strings are anagrams of each other using a frequency counter.",
    """#include <stdbool.h>
bool is_anagram(const char *a, const char *b) {
    int count[256] = {0};
    for (int i = 0; a[i]; i++) count[(unsigned char)a[i]]++;
    for (int i = 0; b[i]; i++) count[(unsigned char)b[i]]--;
    for (int i = 0; i < 256; i++) if (count[i] != 0) return false;
    return true;
}""",
)

_s(
    "Write a C function that finds the intersection of two integer arrays. Returns the count of common elements in result array; does not include duplicates.",
    """int array_intersection(const int *a, int la, const int *b, int lb, int *out) {
    int seen[1024] = {0}, count = 0;
    for (int i = 0; i < la; i++) seen[a[i]] = 1;
    for (int i = 0; i < lb; i++) {
        if (seen[b[i]] == 1) { seen[b[i]] = 2; out[count++] = b[i]; }
    }
    return count;
}""",
)

_s(
    "Write a C function that checks if a string contains all unique characters without additional data structures (bit vector for a-z).",
    """#include <stdbool.h>
bool all_unique(const char *s) {
    int bits = 0;
    for (; *s; s++) {
        int idx = *s - 'a';
        if (idx < 0 || idx > 25) continue;
        if (bits & (1 << idx)) return false;
        bits |= (1 << idx);
    }
    return true;
}""",
)

_s(
    "Write a C function that finds the length of the longest substring without repeating characters using a sliding window and a hash set.",
    """int longest_unique_substr(const char *s) {
    int last[256], max_len = 0, start = 0;
    for (int i = 0; i < 256; i++) last[i] = -1;
    for (int i = 0; s[i]; i++) {
        unsigned char c = (unsigned char)s[i];
        if (last[c] >= start) start = last[c] + 1;
        last[c] = i;
        if (i - start + 1 > max_len) max_len = i - start + 1;
    }
    return max_len;
}""",
)

_s(
    "Write a C function that finds all pairs in a sorted integer array that sum to a given target value. Returns the count of pairs found; stores pairs in output arrays.",
    """int two_sum_all(const int *arr, int len, int target, int *pa, int *pb) {
    int i = 0, j = len - 1, count = 0;
    while (i < j) {
        int sum = arr[i] + arr[j];
        if (sum == target) { pa[count] = arr[i]; pb[count] = arr[j]; count++; i++; j--; }
        else if (sum < target) i++;
        else j--;
    }
    return count;
}""",
)

# ── Memory operations (10) ─────────────────────────────────────────

_s(
    "Write a C function that copies n bytes from source to destination (memcpy) handling overlapping memory correctly.",
    """#include <stddef.h>
void *my_memmove(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d < s) {
        for (size_t i = 0; i < n; i++) d[i] = s[i];
    } else if (d > s) {
        for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
    }
    return dst;
}""",
)

_s(
    "Write a C function that sets n bytes starting at ptr to value c (memset).",
    """#include <stddef.h>
void *my_memset(void *ptr, int c, size_t n) {
    unsigned char *p = (unsigned char *)ptr;
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)c;
    return ptr;
}""",
)

_s(
    "Write a C function that compares two memory regions of length n (memcmp).",
    """#include <stddef.h>
int my_memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *pa = (const unsigned char *)a;
    const unsigned char *pb = (const unsigned char *)b;
    for (size_t i = 0; i < n; i++) {
        if (pa[i] != pb[i]) return (int)pa[i] - (int)pb[i];
    }
    return 0;
}""",
)

_s(
    "Write a C function that copies n bytes from source to destination (memcpy) assuming non-overlapping regions.",
    """#include <stddef.h>
void *my_memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}""",
)

_s(
    "Write a C function that finds the first occurrence of byte c in the first n bytes of s (memchr).",
    """#include <stddef.h>
void *my_memchr(const void *s, int c, size_t n) {
    const unsigned char *p = (const unsigned char *)s;
    for (size_t i = 0; i < n; i++)
        if (p[i] == (unsigned char)c) return (void *)(p + i);
    return 0;
}""",
)

_s(
    "Write a C function that allocates and zeroes memory for an array of count elements of size (calloc).",
    """#include <stdlib.h>
#include <string.h>
void *my_calloc(size_t count, size_t size) {
    void *p = malloc(count * size);
    if (p) memset(p, 0, count * size);
    return p;
}""",
)

_s(
    "Write a C function that swaps the contents of two memory regions of equal size byte-by-byte.",
    """#include <stddef.h>
void memswap(void *a, void *b, size_t n) {
    unsigned char *pa = (unsigned char *)a;
    unsigned char *pb = (unsigned char *)b;
    for (size_t i = 0; i < n; i++) {
        unsigned char t = pa[i]; pa[i] = pb[i]; pb[i] = t;
    }
}""",
)

_s(
    "Write a C function that reverses the order of n bytes in a memory buffer in-place.",
    """#include <stddef.h>
void memrev(void *buf, size_t n) {
    unsigned char *p = (unsigned char *)buf;
    for (size_t i = 0; i < n / 2; i++) {
        unsigned char t = p[i]; p[i] = p[n - 1 - i]; p[n - 1 - i] = t;
    }
}""",
)

_s(
    "Write a C function that finds the longest prefix of n bytes that is all zeroes. Returns the length of the zero prefix.",
    """#include <stddef.h>
size_t zero_prefix(const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    size_t i = 0;
    while (i < n && p[i] == 0) i++;
    return i;
}""",
)

_s(
    "Write a C function that counts the number of zero bytes in a memory buffer.",
    """#include <stddef.h>
size_t count_zero_bytes(const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    size_t c = 0;
    for (size_t i = 0; i < n; i++) if (p[i] == 0) c++;
    return c;
}""",
)

# ── Math / number theory (20) ──────────────────────────────────────

_s(
    "Write a C function that computes the greatest common divisor of two non-negative integers using the Euclidean algorithm.",
    """unsigned int gcd(unsigned int a, unsigned int b) {
    while (b) { unsigned int t = b; b = a % b; a = t; }
    return a;
}""",
)

_s(
    "Write a C function that computes the least common multiple of two unsigned integers.",
    """unsigned int gcd(unsigned int a, unsigned int b) { while (b) { unsigned int t = b; b = a % b; a = t; } return a; }
unsigned int lcm(unsigned int a, unsigned int b) { return a / gcd(a, b) * b; }""",
)

_s(
    "Write a C function that computes factorial of n (n <= 20) using iteration.",
    """unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 2; i <= n; i++) result *= (unsigned long long)i;
    return result;
}""",
)

_s(
    "Write a C function that computes the n-th Fibonacci number using iteration (n >= 0).",
    """unsigned long long fibonacci(int n) {
    if (n <= 1) return (unsigned long long)n;
    unsigned long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) { unsigned long long t = a + b; a = b; b = t; }
    return b;
}""",
)

_s(
    "Write a C function that checks if an unsigned integer is prime using trial division up to sqrt(n).",
    """#include <stdbool.h>
bool is_prime(unsigned int n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (unsigned int d = 3; d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}""",
)

_s(
    "Write a C function that computes integer exponentiation a^b for non-negative b using exponentiation by squaring.",
    """unsigned long long ipow(unsigned long long a, unsigned int b) {
    unsigned long long result = 1;
    while (b) {
        if (b & 1) result *= a;
        a *= a; b >>= 1;
    }
    return result;
}""",
)

_s(
    "Write a C function that computes the integer square root (floor) of a non-negative 64-bit integer using binary search.",
    """#include <stdint.h>
uint32_t isqrt(uint64_t n) {
    if (n <= 1) return (uint32_t)n;
    uint64_t lo = 0, hi = n;
    while (lo <= hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t sq = mid * mid;
        if (sq == n) return (uint32_t)mid;
        if (sq < n) lo = mid + 1; else hi = mid - 1;
    }
    return (uint32_t)hi;
}""",
)

_s(
    "Write a C function that computes a^b mod m for large integers using modular exponentiation.",
    """unsigned long long mod_pow(unsigned long long a, unsigned long long b, unsigned long long m) {
    unsigned long long result = 1;
    a %= m;
    while (b) {
        if (b & 1) result = (result * a) % m;
        a = (a * a) % m; b >>= 1;
    }
    return result;
}""",
)

_s(
    "Write a C function that generates all prime numbers up to n using the Sieve of Eratosthenes. Returns the count. Output array must be pre-allocated with at least n+1 ints.",
    """#include <stdbool.h>
#include <stdlib.h>
int sieve(int n, int *primes_out) {
    if (n < 2) return 0;
    bool *is_prime = (bool *)calloc((size_t)n + 1, sizeof(bool));
    for (int i = 2; i <= n; i++) is_prime[i] = true;
    for (int i = 2; (long long)i * i <= n; i++)
        if (is_prime[i])
            for (int j = i * i; j <= n; j += i) is_prime[j] = false;
    int count = 0;
    for (int i = 2; i <= n; i++) if (is_prime[i]) primes_out[count++] = i;
    free(is_prime);
    return count;
}""",
)

_s(
    "Write a C function that returns the n-th triangular number: Tn = n*(n+1)/2.",
    """unsigned long long triangular(int n) {
    return (unsigned long long)n * ((unsigned long long)n + 1) / 2;
}""",
)

_s(
    "Write a C function that checks if a non-negative integer is a perfect square.",
    """#include <stdbool.h>
bool is_perfect_square(int n) {
    if (n < 0) return false;
    int r = 0;
    for (int i = 0; i * i <= n; i++) { r = i; }
    return r * r == n;
}""",
)

_s(
    "Write a C function that computes the sum of proper divisors of a positive integer.",
    """int sum_divisors(int n) {
    if (n <= 1) return 0;
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) { sum += i; if (i != n / i) sum += n / i; }
    }
    return sum;
}""",
)

_s(
    "Write a C function that evaluates a quadratic polynomial ax^2 + bx + c for given integer coefficients and x.",
    """long long quadratic(int a, int b, int c, int x) {
    long long ax2 = (long long)a * x * x;
    return ax2 + (long long)b * x + c;
}""",
)

_s(
    "Write a C function that computes the digital root of a non-negative integer (repeated digit sum until single digit).",
    """int digital_root(int n) {
    return n == 0 ? 0 : 1 + (n - 1) % 9;
}""",
)

_s(
    "Write a C function that computes the number of ways to climb n stairs taking 1 or 2 steps at a time (like Fibonacci, n >= 0).",
    """int climb_stairs(int n) {
    if (n <= 2) return n > 0 ? n : 1;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++) { int t = a + b; a = b; b = t; }
    return b;
}""",
)

_s(
    "Write a C function that generates the first n rows of Pascal's triangle. Stores values in a flat pre-allocated array; the k-th entry of row r (0-indexed) is at index r*(r+1)/2 + k. Returns total number of entries written.",
    """int pascals_triangle(int n, int *out) {
    int w = 0;
    for (int r = 0; r < n; r++) {
        for (int k = 0; k <= r; k++) {
            if (k == 0 || k == r) out[w++] = 1;
            else out[w++] = out[w - r - 1] + out[w - r];
        }
    }
    return w;
}""",
)

_s(
    "Write a C function that converts a non-negative integer to its string representation in base 10 (itoa).",
    """void itoa(int n, char *buf) {
    int i = 0;
    do { buf[i++] = (char)('0' + n % 10); n /= 10; } while (n);
    buf[i] = '\\0';
    for (int j = 0; j < i / 2; j++) { char t = buf[j]; buf[j] = buf[i-1-j]; buf[i-1-j] = t; }
}""",
)

_s(
    "Write a C function that converts a string representing a non-negative integer in base 10 to int (atoi).",
    """int atoi_simple(const char *s) {
    int result = 0;
    while (*s >= '0' && *s <= '9') {
        result = result * 10 + (*s - '0'); s++;
    }
    return result;
}""",
)

_s(
    "Write a C function that rounds a positive double to the nearest integer using banker's rounding (round half to even).",
    """#include <math.h>
double round_banker(double x) {
    double int_part;
    double frac = modf(x, &int_part);
    if (frac > 0.5) return int_part + 1.0;
    if (frac < 0.5) return int_part;
    int ip = (int)int_part;
    return (ip & 1) ? int_part + 1.0 : int_part;
}""",
)

_s(
    "Write a C function that computes the sum of first n natural numbers using the closed-form formula.",
    """unsigned long long sum_n(int n) {
    return (unsigned long long)n * (unsigned long long)(n + 1) / 2;
}""",
)

# ── Recursion and DP (10) ──────────────────────────────────────────

_s(
    "Write a C function that computes the number of combinations C(n,k) using a multiplicative formula that avoids overflow for moderate n.",
    """unsigned long long n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    unsigned long long result = 1;
    for (int i = 0; i < k; i++) {
        result = result * (unsigned long long)(n - i) / (unsigned long long)(i + 1);
    }
    return result;
}""",
)

_s(
    "Write a C function that computes the maximum value achievable in the 0/1 knapsack problem given integer weights and values. Uses bottom-up DP with a 1D array.",
    """int knapsack(int cap, const int *wt, const int *val, int n) {
    int dp[256] = {0};
    for (int i = 0; i < n; i++)
        for (int w = cap; w >= wt[i]; w--)
            if (dp[w - wt[i]] + val[i] > dp[w])
                dp[w] = dp[w - wt[i]] + val[i];
    return dp[cap];
}""",
)

_s(
    "Write a C function that computes the number of ways to make change for amount using coins of given denominations (unbounded knapsack). Returns count using 1D DP.",
    """int coin_change_ways(int amount, const int *coins, int n) {
    int dp[256] = {0}; dp[0] = 1;
    for (int i = 0; i < n; i++)
        for (int a = coins[i]; a <= amount; a++)
            dp[a] += dp[a - coins[i]];
    return dp[amount];
}""",
)

_s(
    "Write a C function that finds the length of the longest common subsequence (LCS) of two strings using 2D DP.",
    """int max2(int a, int b) { return a > b ? a : b; }
int lcs_length(const char *a, const char *b) {
    int la = 0, lb = 0;
    while (a[la]) la++; while (b[lb]) lb++;
    int dp[101][101] = {{0}};
    for (int i = 1; i <= la; i++)
        for (int j = 1; j <= lb; j++)
            dp[i][j] = (a[i-1] == b[j-1])
                ? dp[i-1][j-1] + 1
                : max2(dp[i-1][j], dp[i][j-1]);
    return dp[la][lb];
}""",
)

_s(
    "Write a C function that computes the edit distance (Levenshtein) between two strings using optimized DP with O(min(m,n)) space.",
    """#include <string.h>
int min3_i(int a, int b, int c) { if (a <= b && a <= c) return a; if (b <= c) return b; return c; }
int edit_distance(const char *a, const char *b) {
    int m = (int)strlen(a), n = (int)strlen(b);
    int prev[256], cur[256];
    for (int j = 0; j <= n; j++) prev[j] = j;
    for (int i = 1; i <= m; i++) {
        cur[0] = i;
        for (int j = 1; j <= n; j++)
            cur[j] = min3_i(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (a[i-1] != b[j-1]));
        for (int j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}""",
)

_s(
    "Write a C function that computes the longest increasing subsequence (LIS) length of an integer array using patience sorting (binary search on tails).",
    """int lis_length(const int *arr, int len) {
    int tails[256], sz = 0;
    for (int i = 0; i < len; i++) {
        int lo = 0, hi = sz;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (tails[mid] < arr[i]) lo = mid + 1; else hi = mid;
        }
        tails[lo] = arr[i];
        if (lo == sz) sz++;
    }
    return sz;
}""",
)

_s(
    "Write a C function that implements binary exponentiation to compute Fibonacci number Fn in O(log n) using matrix multiplication.",
    """unsigned long long fib_matrix(int n) {
    if (n == 0) return 0;
    unsigned long long a = 0, b = 1, c = 1, d = 1;
    unsigned long long ra = 1, rb = 0, rc = 0, rd = 1;
    n--;
    while (n) {
        if (n & 1) {
            unsigned long long na = ra * a + rb * c, nb = ra * b + rb * d;
            unsigned long long nc = rc * a + rd * c, nd = rc * b + rd * d;
            ra = na; rb = nb; rc = nc; rd = nd;
        }
        unsigned long long na = a * a + b * c, nb = a * b + b * d;
        unsigned long long nc = c * a + d * c, nd = c * b + d * d;
        a = na; b = nb; c = nc; d = nd;
        n >>= 1;
    }
    return rb;
}""",
)

_s(
    "Write a C function that solves the rod-cutting problem: given an array of prices for lengths 1..n, find max revenue using 1D DP.",
    """int rod_cutting(const int *price, int n) {
    int dp[256] = {0};
    for (int i = 1; i <= n; i++) {
        int best = price[i-1];
        for (int j = 1; j < i; j++)
            if (dp[i - j] + price[j-1] > best) best = dp[i - j] + price[j-1];
        dp[i] = best;
    }
    return dp[n];
}""",
)

_s(
    "Write a C function that computes the minimum cost path from top-left to bottom-right of a grid (moving only right or down). Uses 1D DP.",
    """int min_path_sum(const int *grid, int rows, int cols) {
    int dp[256];
    dp[0] = grid[0];
    for (int c = 1; c < cols; c++) dp[c] = dp[c - 1] + grid[c];
    for (int r = 1; r < rows; r++) {
        dp[0] += grid[r * cols];
        for (int c = 1; c < cols; c++)
            dp[c] = grid[r * cols + c] + (dp[c - 1] < dp[c] ? dp[c - 1] : dp[c]);
    }
    return dp[cols - 1];
}""",
)

_s(
    "Write a C function that implements the Tower of Hanoi solver, moving n disks from source to target using auxiliary peg. Writes moves into a pre-allocated array of move strings (format: \"1->3\"). Returns the number of moves (2^n - 1).",
    """int hanoi_moves(int n, int src, int tgt, int aux, int moves[][2], int *pos) {
    if (n == 0) return 0;
    int c = 0;
    c += hanoi_moves(n - 1, src, aux, tgt, moves, pos);
    moves[*pos][0] = src; moves[*pos][1] = tgt; (*pos)++; c++;
    c += hanoi_moves(n - 1, aux, tgt, src, moves, pos);
    return c;
}""",
)

# ── Graphs (10) ────────────────────────────────────────────────────

_s(
    "Write a C function that performs Breadth-First Search (BFS) on a graph represented as an adjacency matrix. Visits nodes in BFS order starting from node 0, calling a callback for each visited node.",
    """#define MAX_NODES 64
typedef void (*NodeVisitor)(int);
void bfs_matrix(int graph[MAX_NODES][MAX_NODES], int n, NodeVisitor visit) {
    int visited[MAX_NODES] = {0};
    int q[MAX_NODES], head = 0, tail = 0;
    q[tail++] = 0; visited[0] = 1;
    while (head < tail) {
        int v = q[head++];
        visit(v);
        for (int w = 0; w < n; w++)
            if (graph[v][w] && !visited[w]) { visited[w] = 1; q[tail++] = w; }
    }
}""",
)

_s(
    "Write a C function that performs Depth-First Search (DFS) iteratively on a graph represented as an adjacency matrix, using an explicit stack.",
    """#define MAX_NODES 64
typedef void (*NodeVisitor)(int);
void dfs_matrix(int graph[MAX_NODES][MAX_NODES], int n, NodeVisitor visit) {
    int visited[MAX_NODES] = {0};
    int stack[MAX_NODES], top = -1;
    stack[++top] = 0;
    while (top >= 0) {
        int v = stack[top--];
        if (visited[v]) continue;
        visited[v] = 1; visit(v);
        for (int w = n - 1; w >= 0; w--)
            if (graph[v][w] && !visited[w]) stack[++top] = w;
    }
}""",
)

_s(
    "Write a C function that detects a cycle in an undirected graph represented as an adjacency matrix using DFS.",
    """#include <stdbool.h>
#define MAX_NODES 64
bool dfs_cycle(int g[MAX_NODES][MAX_NODES], int v, int parent, int *visited, int n) {
    visited[v] = 1;
    for (int w = 0; w < n; w++) {
        if (!g[v][w]) continue;
        if (!visited[w]) { if (dfs_cycle(g, w, v, visited, n)) return true; }
        else if (w != parent) return true;
    }
    return false;
}
bool has_cycle_undirected(int graph[MAX_NODES][MAX_NODES], int n) {
    int visited[MAX_NODES] = {0};
    for (int v = 0; v < n; v++)
        if (!visited[v] && dfs_cycle(graph, v, -1, visited, n)) return true;
    return false;
}""",
)

_s(
    "Write a C function that performs a topological sort on a directed acyclic graph represented as an adjacency matrix using DFS. Returns 1 on success (writes order to output), 0 if a cycle is detected.",
    """#include <stdbool.h>
#define MAX_NODES 64
int toposort_dfs(int g[MAX_NODES][MAX_NODES], int n, int *order) {
    int state[MAX_NODES] = {0};
    int pos = n;
    bool ok = true;
    void dfs(int v) {
        state[v] = 1;
        for (int w = 0; w < n; w++) {
            if (!g[v][w]) continue;
            if (state[w] == 1) { ok = false; return; }
            if (state[w] == 0) dfs(w);
        }
        state[v] = 2;
        order[--pos] = v;
    }
    for (int v = 0; v < n; v++) if (state[v] == 0) dfs(v);
    return ok ? 1 : 0;
}""",
)

_s(
    "Write a C function that implements Dijkstra's algorithm for shortest paths from a source node in a graph with non-negative edge weights, represented as an adjacency matrix.",
    """#include <limits.h>
#include <stdbool.h>
#define MAX_NODES 64
void dijkstra(int g[MAX_NODES][MAX_NODES], int n, int src, int *dist) {
    bool done[MAX_NODES] = {false};
    for (int i = 0; i < n; i++) dist[i] = INT_MAX;
    dist[src] = 0;
    for (int _ = 0; _ < n; _++) {
        int u = -1, best = INT_MAX;
        for (int i = 0; i < n; i++)
            if (!done[i] && dist[i] < best) { best = dist[i]; u = i; }
        if (u < 0) break;
        done[u] = true;
        for (int v = 0; v < n; v++)
            if (g[u][v] && !done[v] && dist[u] + g[u][v] < dist[v])
                dist[v] = dist[u] + g[u][v];
    }
}""",
)

_s(
    "Write a C function that implements the Floyd-Warshall all-pairs shortest path algorithm on a directed graph represented as an adjacency matrix with edge weights. Uses INT_MAX for no edge.",
    """#include <limits.h>
#define MAX_NODES 64
void floyd_warshall(int g[MAX_NODES][MAX_NODES], int n) {
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (g[i][k] != INT_MAX && g[k][j] != INT_MAX &&
                    g[i][k] + g[k][j] < g[i][j])
                    g[i][j] = g[i][k] + g[k][j];
}""",
)

_s(
    "Write a C function that computes the connected components of an undirected graph represented as an adjacency matrix. Returns the component label for each node in the output array. Uses BFS.",
    """#define MAX_NODES 64
int connected_components(int g[MAX_NODES][MAX_NODES], int n, int *comp) {
    for (int i = 0; i < n; i++) comp[i] = -1;
    int label = 0, q[MAX_NODES], head, tail;
    for (int v = 0; v < n; v++) {
        if (comp[v] >= 0) continue;
        head = tail = 0; q[tail++] = v; comp[v] = label;
        while (head < tail) {
            int u = q[head++];
            for (int w = 0; w < n; w++)
                if (g[u][w] && comp[w] < 0) { comp[w] = label; q[tail++] = w; }
        }
        label++;
    }
    return label;
}""",
)

_s(
    "Write a C function that performs flood fill on a 2D grid: starting from (sr, sc), fills connected cells matching the original color with a new color (4-directional).",
    """#define MAX_RC 64
void flood_fill(int grid[MAX_RC][MAX_RC], int rows, int cols, int sr, int sc, int new_color) {
    int orig = grid[sr][sc];
    if (orig == new_color) return;
    int qr[MAX_RC * MAX_RC], qc[MAX_RC * MAX_RC], head = 0, tail = 0;
    qr[tail] = sr; qc[tail] = sc; tail++;
    grid[sr][sc] = new_color;
    int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
    while (head < tail) {
        int r = qr[head], c = qc[head]; head++;
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == orig) {
                grid[nr][nc] = new_color;
                qr[tail] = nr; qc[tail] = nc; tail++;
            }
        }
    }
}""",
)

_s(
    "Write a C function that counts the number of islands in a 2D binary grid, where 1 is land and 0 is water. An island is a 4-directionally connected group of 1s.",
    """#define MAX_RC 64
void dfs_island(int g[MAX_RC][MAX_RC], int r, int c, int rows, int cols) {
    if (r < 0 || r >= rows || c < 0 || c >= cols || g[r][c] != 1) return;
    g[r][c] = 0;
    dfs_island(g, r - 1, c, rows, cols);
    dfs_island(g, r + 1, c, rows, cols);
    dfs_island(g, r, c - 1, rows, cols);
    dfs_island(g, r, c + 1, rows, cols);
}
int count_islands(int g[MAX_RC][MAX_RC], int rows, int cols) {
    int count = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (g[r][c] == 1) { count++; dfs_island(g, r, c, rows, cols); }
    return count;
}""",
)

_s(
    "Write a C function that implements Kruskal's algorithm for Minimum Spanning Tree using a simple Union-Find. Edges are given as triples (u, v, weight) in a flat array; n edges, n_nodes vertices. Returns MST total weight or -1 if disconnected.",
    """#define MAX_V 64
int uf_parent[MAX_V];
int uf_find(int x) {
    while (uf_parent[x] != x) { uf_parent[x] = uf_parent[uf_parent[x]]; x = uf_parent[x]; }
    return x;
}
void uf_union(int a, int b) { uf_parent[uf_find(a)] = uf_find(b); }
void sort_edges(int *edges, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - 1 - i; j++)
            if (edges[3*j+2] > edges[3*(j+1)+2])
                for (int k = 0; k < 3; k++) { int t = edges[3*j+k]; edges[3*j+k] = edges[3*(j+1)+k]; edges[3*(j+1)+k] = t; }
}
int kruskal(int *edges, int n_edges, int n_nodes) {
    for (int i = 0; i < n_nodes; i++) uf_parent[i] = i;
    sort_edges(edges, n_edges);
    int total = 0, used = 0;
    for (int i = 0; i < n_edges; i++) {
        int u = edges[3*i], v = edges[3*i+1], w = edges[3*i+2];
        if (uf_find(u) != uf_find(v)) { uf_union(u, v); total += w; used++; }
    }
    return (used == n_nodes - 1) ? total : -1;
}""",
)

# ── Misc (10) ─────────────────────────────────────────────────────

_s(
    "Write a C function that converts a temperature in Celsius to Fahrenheit.",
    """double celsius_to_fahrenheit(double c) { return c * 9.0 / 5.0 + 32.0; }""",
)

_s(
    "Write a C function that computes the area of a circle given its radius.",
    """double circle_area(double r) { return 3.14159265358979323846 * r * r; }""",
)

_s(
    "Write a C function that swaps the values of two integers using a temporary variable.",
    """void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }""",
)

_s(
    "Write a C function that implements a simple linear congruential pseudorandom number generator. Maintains state via a pointer to the seed.",
    """int lcg_rand(unsigned int *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (int)(*seed >> 16);
}""",
)

_s(
    "Write a C function that computes the dot product of two integer vectors of length n.",
    """long long dot_product(const int *a, const int *b, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) sum += (long long)a[i] * (long long)b[i];
    return sum;
}""",
)

_s(
    "Write a C function that computes the cross product of two 3D vectors (as arrays of 3 doubles) and stores the result in a third array.",
    """void cross_product(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}""",
)

_s(
    "Write a C function that checks if a year is a leap year according to Gregorian calendar rules.",
    """#include <stdbool.h>
bool is_leap_year(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}""",
)

_s(
    "Write a C function that computes the day of the week for a given date using Zeller's congruence. Returns 0=Saturday .. 6=Friday.",
    """int zeller(int d, int m, int y) {
    if (m < 3) { m += 12; y--; }
    int k = y % 100, j = y / 100;
    return (d + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7;
}""",
)

_s(
    "Write a C function that runs a basic for-loop summing integers from 1 to n and returns the sum.",
    """int sum_1_to_n(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += i;
    return s;
}""",
)

_s(
    "Write a C function that runs a basic while-loop counting how many times n can be divided by 2 before reaching 0.",
    """int count_halves(int n) {
    int c = 0;
    while (n > 0) { n /= 2; c++; }
    return c;
}""",
)


# ── Build logic ─────────────────────────────────────────────────────

def compile_c_to_asm(c_code: str) -> Optional[str]:
    """Compile C code to x86-64 assembly (Intel syntax, -O2)."""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(c_code)
        c_path = f.name
    try:
        result = subprocess.run(
            ["gcc", "-O2", "-S", "-masm=intel", "-fno-asynchronous-unwind-tables",
             "-o", "-", c_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  [WARN] GCC -S failed: {result.stderr[:200]}")
            return None
        # Strip .file and .ident directives — GCC embeds temp file paths.
        lines = result.stdout.splitlines()
        clean = [l for l in lines if not l.lstrip().startswith(('.file', '.ident'))]
        return '\n'.join(clean) + '\n'
    finally:
        os.unlink(c_path)


def verify_asm_compiles(asm_code: str) -> bool:
    """Verify that the assembly assembles with gcc -c."""
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_code)
        s_path = f.name
    try:
        result = subprocess.run(
            ["gcc", "-c", s_path, "-o", os.devnull],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    finally:
        os.unlink(s_path)


def check_gcc() -> bool:
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description="Build NL-spec->C->asm triples")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no GCC")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--count", type=int, default=0, help="Max count (0 = all)")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[dry-run] {len(SPECS)} specs defined")
        for s in SPECS[:3]:
            print(f"  spec: {s['spec'][:80]}...")
            print(f"  code: {len(s['c_code'])} chars, first line: {s['c_code'].split(chr(10))[0][:60]}...")
        print("[dry-run] OK")
        return

    if not check_gcc():
        sys.exit("GCC not found. Install gcc or use --dry-run.")

    out_dir = Path("data/nl_asm")
    out_dir.mkdir(parents=True, exist_ok=True)

    triples_path = out_dir / "nl_spec_triples.jsonl"
    srpo_path = out_dir / "srpo_nl_train.jsonl"

    start = args.start
    end = start + args.count if args.count > 0 else len(SPECS)
    spec_slice = SPECS[start:end]

    verified_count = 0
    triples_count = 0
    with open(triples_path, "w") as ft, open(srpo_path, "w") as fs:
        for i, s in enumerate(spec_slice):
            idx = start + i
            c_code = s["c_code"]
            print(f"[{idx + 1}/{len(SPECS)}] {s['spec'][:70]}...")

            asm = compile_c_to_asm(c_code)
            verified = verify_asm_compiles(asm) if asm else False
            verified_flag = verified and asm is not None

            if asm is None:
                print(f"  SKIP (GCC -S failed)")
                continue

            triple = {
                "spec": s["spec"],
                "c_code": c_code,
                "asm": asm,
                "verified": verified_flag,
            }
            ft.write(json.dumps(triple, ensure_ascii=False) + "\n")
            triples_count += 1
            if verified_flag:
                verified_count += 1

            srpo_rec = {
                "prompt": f"Write C code and explain the resulting assembly for: {s['spec']}",
                "target": f"C:\n```c\n{c_code}\n```\n\nAssembly (x86-64, GCC -O2):\n```asm\n{asm}\n```",
            }
            fs.write(json.dumps(srpo_rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {triples_count} triples to {triples_path}")
    print(f"Saved {triples_count} SRPO records to {srpo_path}")
    print(f"Verification: {verified_count}/{triples_count} assembly roundtrip passed")


if __name__ == "__main__":
    main()
