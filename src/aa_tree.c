#include<stdio.h>
#include<stdlib.h>
#include"aa_tree.h"

static
int64_t comp_aatree_val(aatree_val a, aatree_val b)
{
  return a - b;
}

static
aatree_node *skew(aatree_node *tree)
{
  if (tree == &nil){
    return tree;
  }
  if (tree->left->level == tree->level){
    aatree_node *l;
    l = tree->left;
    tree->left = l->right;
    l->right = tree;
    return l;
  }
  return tree;
}

static
aatree_node *split(aatree_node *tree)
{
  if (tree == &nil){
    return tree;
  }
  if (tree->level == tree->right->right->level){
    aatree_node *r;
    r = tree->right;
    tree->right = r->left;
    r->left = tree;
    r->level++;
    return r;
  }
  return tree;
}

static
aatree_node *alloc_node(aatree_val x)
{
  aatree_node *ret;
  ret = (aatree_node*)malloc(sizeof(aatree_node));
  if (ret == NULL){
    return NULL;
  }
  ret->val = x;
  ret->level = 1;
  ret->left = &nil;
  ret->right = &nil;
  return ret;
}

/* flagは値の挿入が起こると1，すでに値が入っていたら0 */
aatree_node *__aa_tree_insert__(aatree_val x, aatree_node *tree, int64_t *flag)
{
  int64_t cmp;
  *flag = 1;
  if (tree == &nil){
    aatree_node *ret;
    ret = alloc_node(x);
    if (ret == NULL){
      fprintf(stderr,"failed to alloc tree node\n");
    }
    return ret;
  }
  cmp = comp_aatree_val(x, tree->val);
  if (cmp < 0){
    tree->left = __aa_tree_insert__(x, tree->left, flag);
  }else if (cmp > 0){
    tree->right = __aa_tree_insert__(x, tree->right, flag);
  }else{
    *flag = 0;
    return tree;//すでに木に入っている
  }
  tree = skew(tree);
  tree = split(tree);
  return tree;
}

aatree_node *aa_tree_insert(aatree_val x, aatree_node *tree, int64_t *flag)
{
  if(tree == NULL){
    tree = &nil;
  }
  return __aa_tree_insert__(x, tree, flag);
}

/* flagはdeleteが成功すると1失敗すると0 */
static
aatree_node *__aa_tree_delete__(struct __del_info__ *info, aatree_node *tree, int64_t *flag)
{
  int64_t cmp;
  if (tree == &nil){
    return tree;
  }
  info->last = tree;
  cmp = comp_aatree_val(info->val, tree->val);

  if (cmp < 0){
    tree->left = __aa_tree_delete__(info, tree->left, flag);
  }else{
    info->del = tree;
    tree->right = __aa_tree_delete__(info, tree->right, flag);
  }
  if (tree == info->last && info->del != &nil && comp_aatree_val(info->val, info->del->val) == 0){
    aatree_node *r;
    r = tree->right;
    info->del->val = tree->val;
    info->del = &nil;
    free(tree);
    *flag = 1;
    return r;
  }
  if (tree->left->level < tree->level - 1 || tree->right->level < tree->level - 1){
    tree->level--;
    if(tree->right->level > tree->level){
      tree->right->level = tree->level;
    }
    tree = skew(tree);
    tree->right = skew(tree->right);
    tree->right->right = skew(tree->right->right);
    tree = split(tree);
    tree->right = split(tree->right);
  }
  return tree;
}

aatree_node *aa_tree_delete(aatree_val x, aatree_node *tree, int64_t *flag)
{
  struct __del_info__ info;
  info.val = x;
  info.last = &nil;
  info.del = &nil;

  *flag = 0;

  if(tree != NULL){
    tree = __aa_tree_delete__(&info, tree, flag);
    if(tree == &nil){
      tree = NULL;
    }
  }
  return tree;
}

void aa_tree_free(aatree_node *tree)
{
  if(tree != NULL && tree != &nil){
    aa_tree_free(tree->left);
    aa_tree_free(tree->right);
    free(tree);
  }
}

void aa_tree_print(aatree_node *tree){
  if (tree == NULL || tree == &nil){
    return;
  }
  aa_tree_print(tree->left);
  printf("%lx\n", tree->val);
  aa_tree_print(tree->right);
}
static
void __aatree2vec__(aatree_node *tree, aatree_val **dst){
  if (tree == NULL || tree == &nil){
    return;
  }
  __aatree2vec__(tree->left, dst);
  *(*dst)++ = tree->val;
  __aatree2vec__(tree->right, dst);
}

void aa_tree2vec(aatree_node *tree, aatree_val *dst){
  if (tree == NULL || tree == &nil){
    return;
  }
  __aatree2vec__(tree, &dst);
}

int aa_tree_search(aatree_node *tree, aatree_val x){
  int64_t cmp;
  while (tree && tree != &nil) {
    cmp = comp_aatree_val(x, tree->val);
    if (cmp == 0) {
      return 1;
    } else if (cmp < 0) {
      tree = tree->left;
    } else {
      tree = tree->right;
    }
  }
  return 0;
}
