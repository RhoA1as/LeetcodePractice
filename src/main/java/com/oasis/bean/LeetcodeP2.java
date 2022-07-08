package com.oasis.bean;

import java.util.List;

public class LeetcodeP2 {

    //https://leetcode-cn.com/problems/construct-quad-tree/ 建立四叉树
    int[][] mPreSum;
    int[][] mGrid;
    public Node construct(int[][] grid) {
        int m = grid.length;
        int[][] preSum = new int[m+1][m+1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                preSum[i][j] = preSum[i-1][j] + preSum[i][j-1] - preSum[i-1][j-1] + grid[i-1][j-1];
            }
        }
        mPreSum = preSum;
        mGrid = grid;
        return dfsConstruct(0, 0, m, m);
    }

    public Node dfsConstruct(int r0, int c0, int r1, int c1){
        int sum = getSum(r0, c0, r1, c1);
        if(sum == 0){
            return new Node(false, true);
        }
        if(sum == (r1 - r0) * (c1 - c0)){
            return new Node(true, true);
        }
        int r = (r0 + r1) / 2, c = (c0 + c1) / 2;
        return new Node(true, false,
                dfsConstruct(r0, c0, r, c),
                dfsConstruct(r0, c, r, c1),
                dfsConstruct(r, c0, r1, c),
                dfsConstruct(r, c, r1, c1)
                );
    }

    public int getSum(int r0, int c0, int r1, int c1){
        return mPreSum[r1][c1] - mPreSum[r0][c1] - mPreSum[r1][c0] + mPreSum[r0][c0];
    }

    //https://leetcode.cn/problems/replace-words/ 单词替换
    class TriNode{
        boolean isEnd;
        TriNode[] children = new TriNode[26];
    }

    TriNode root = new TriNode();

    private void add(String s){
        TriNode tmp = root;
        for (int i = 0; i < s.length(); i++) {
            int c = s.charAt(i) - 'a';
            if(tmp.children[c] == null) tmp.children[c] = new TriNode();
            tmp = tmp.children[c];
        }
        tmp.isEnd = true;
    }

    private String query(String s){
        TriNode tmp = root;
        for (int i = 0; i < s.length(); i++) {
            int c = s.charAt(i) - 'a';
            if(tmp.children[c] == null) break;
            if(tmp.children[c].isEnd) return s.substring(0, i + 1);
            tmp = tmp.children[c];
        }
        return s;
    }
    public String replaceWords(List<String> dictionary, String sentence) {
        for (String s : dictionary) {
            add(s);
        }
        StringBuilder ans = new StringBuilder();
        String[] s = sentence.split(" ");
        int n = s.length;
        for (int i = 0; i < n; i++) {
            ans.append(query(s[i]));
            if(i != n-1) ans.append(" ");
        }
        return ans.toString();
    }
}
class Node {
    public boolean val;
    public boolean isLeaf;
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;


    public Node() {
        this.val = false;
        this.isLeaf = false;
        this.topLeft = null;
        this.topRight = null;
        this.bottomLeft = null;
        this.bottomRight = null;
    }

    public Node(boolean val, boolean isLeaf) {
        this.val = val;
        this.isLeaf = isLeaf;
        this.topLeft = null;
        this.topRight = null;
        this.bottomLeft = null;
        this.bottomRight = null;
    }

    public Node(boolean val, boolean isLeaf, Node topLeft, Node topRight, Node bottomLeft, Node bottomRight) {
        this.val = val;
        this.isLeaf = isLeaf;
        this.topLeft = topLeft;
        this.topRight = topRight;
        this.bottomLeft = bottomLeft;
        this.bottomRight = bottomRight;
    }
}