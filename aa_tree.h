#include<stdint.h>
// taken from https://github.com/h-hayakawa/aa-tree
// it was the first one i found that worked without it taking forever to adjust

/* int以外を使いたかったらここから改造 */
typedef uint64_t aatree_val;

typedef struct __aatree_node__{
      int32_t level; /* AA-tree制御用変数 */
        aatree_val val; /* 格納してる値本体 */
          struct __aatree_node__ *left;
            struct __aatree_node__ *right;
} aatree_node;

/* 値削除用制御変数 */
struct __del_info__{
      aatree_val val;
        aatree_node *last;
          aatree_node *del;
};

/* nilを定義せずNULLを使うとロクなことがない */
static aatree_node nil = {0, 0, &nil, &nil};

/* xが木の中に無ければ追加、flagに1をセット、あれば何もせず、flagに0をセット、操作後の木のrootを返す */
aatree_node *aa_tree_insert(aatree_val x, aatree_node *tree, int64_t *flag);

/* xが木の中にあれば削除、flagに1をセット、無ければ何もせず、flagに0をセット、操作後の木のrootを返す */
aatree_node *aa_tree_delete(aatree_val x, aatree_node *tree, int64_t *flag);

/* お掃除 */
void aa_tree_free(aatree_node *tree);

/* 木を舐めながら印字、結果はソートされた列になる */
void aa_tree_print(aatree_node *tree);

/* 木構造の中身を配列に移す、木を舐めるので自動的にソートされた結果になる */
void aa_tree2vec(aatree_node *tree, aatree_val *dst);

/* aatree_valが木の中にあるかどうか、あれば1、なければ0 */
int aa_tree_search(aatree_node *tree, aatree_val x);
