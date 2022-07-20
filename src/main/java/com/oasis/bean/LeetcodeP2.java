package com.oasis.bean;

import org.junit.jupiter.api.Test;

import java.util.*;

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

    //https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/ 最长的斐波那契子序列的长度
    public int lenLongestFibSubseq(int[] arr) {
        int n = arr.length, ans = 0;
        int[][] dp = new int[n][n];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map.put(arr[i], i);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i - 1; j >= 0 && j + 2 > ans; j--) {
                int div = arr[i] - arr[j];
                if(div >= arr[j]) break;
                int k = map.getOrDefault(div, -1);
                if(k == -1) continue;
                dp[i][j] = Math.max(3, dp[j][k] + 1);
                ans = Math.max(dp[i][j], ans);
            }
        }
        return ans;
    }

    //https://leetcode.cn/problems/asteroid-collision/ 行星碰撞
    public int[] asteroidCollision(int[] asteroids) {
        if(asteroids == null || asteroids.length == 0){
            return asteroids;
        }
        Deque<Integer> stack = new ArrayDeque<>();
        for (int asteroid : asteroids) {
            boolean isLive = true;
            while (isLive && !stack.isEmpty() && stack.peekLast() > 0 && asteroid < 0){
                int top = stack.peekLast();
                if(top >= -asteroid) isLive = false;
                if(top <= -asteroid) stack.pollLast();
            }
            if(isLive) stack.offer(asteroid);
        }
        int size = stack.size(), i = 0;
        int[] ans = new int[size];
        for (int a : stack) {
            ans[i++] = a;
        }
        return ans;
    }

    //https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix/ 奇数值单元格的数目
    public int oddCells(int m, int n, int[][] indices) {
        long row = 0, col = 0;
        for (int[] index : indices) {
            row ^= 1L << index[0];
            col ^= 1L << index[1];
        }
        int row_cnt = Long.bitCount(row), col_cnt = Long.bitCount(col);
        return row_cnt * (n - col_cnt) + (m - row_cnt) * col_cnt;
    }

    @Test
    public void test(){
        MagicDictionary magicDictionary = new MagicDictionary();
        magicDictionary.buildDict(new String[]{"hello", "hallo", "leetcode"});
        System.out.println(magicDictionary.search("hello"));
        System.out.println(magicDictionary.search("hhllo"));
        System.out.println(magicDictionary.search("hell"));
        System.out.println(magicDictionary.search("leetcoded"));
    }

    //https://leetcode.cn/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/ 四叉树交集
    public Node intersect(Node quadTree1, Node quadTree2) {
        if(quadTree1.isLeaf){
            if(quadTree1.val){
                return new Node(true, true);
            }
            return quadTree2;
        }
        if(quadTree2.isLeaf){
            return intersect(quadTree2, quadTree1);
        }
        Node topLeft = intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node topRight = intersect(quadTree1.topRight, quadTree2.topRight);
        Node bottomLeft = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node bottomRight = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        boolean a = topLeft.isLeaf && topRight.isLeaf && bottomLeft.isLeaf && bottomRight.isLeaf;
        boolean b = topLeft.val && topRight.val && bottomLeft.val && bottomRight.val;
        boolean c = topLeft.val || topRight.val || bottomLeft.val || bottomRight.val;
        if(a && (b || !c)){
            return new Node(topLeft.val, true);
        }
        return new Node(false, false, topLeft, topRight, bottomLeft, bottomRight);
    }

    //https://leetcode.cn/problems/qIsx9U/ 滑动窗口的平均值
    class MovingAverage {

        private int mSize;

        private double mSum;

        private Deque<Integer> mqueue;

        /** Initialize your data structure here. */
        public MovingAverage(int size) {
            mSize = size;
            mqueue = new ArrayDeque<>();
        }

        public double next(int val) {
            if(mqueue.size() == mSize) mSum -= mqueue.pollFirst();
            mqueue.offer(val);
            mSum += val;
            return mSum / mqueue.size();
        }
    }
}


//https://leetcode.cn/problems/implement-magic-dictionary/ 实现一个魔法字典
class MagicDictionary {

    class TriNode{
        boolean isEnd;
        TriNode[] children = new TriNode[26];
    }

    TriNode root;

    private void add(String s){
        TriNode tmp = root;
        for (int i = 0; i < s.length(); i++) {
            int c = s.charAt(i) - 'a';
            if(tmp.children[c] == null){
                tmp.children[c] = new TriNode();
            }
            tmp = tmp.children[c];
        }
        tmp.isEnd = true;
    }

    public MagicDictionary() {
        root = new TriNode();
    }

    public void buildDict(String[] dictionary) {
        for (String s : dictionary) {
            add(s);
        }
    }

    public boolean search(String searchWord) {
        return searchImpl(searchWord, 0, 1, root);
    }

    private boolean searchImpl(String s, int idx, int limit, TriNode cur){
        TriNode tmp = cur;
        for (int i = idx; i < s.length(); i++) {
            int c = s.charAt(i) - 'a';
            if(limit != 0) {
                for (int j = 0; j < 26; j++) {
                    if (tmp.children[j] == null) continue;
                    int nxt_limit = c == j ? limit : limit - 1;
                    if (searchImpl(s, i + 1, nxt_limit, tmp.children[j])) {
                        return true;
                    }
                }
                return false;
            } else if(tmp.children[c] != null){
                tmp = tmp.children[c];
            } else {
                return false;
            }
        }
        return limit == 0 && tmp.isEnd;
    }

    @Test
    public void test3(){
        arrayNesting(new int[]{5, 4, 0, 3, 1, 6, 2});
    }

    //https://leetcode.cn/problems/array-nesting/ 数组嵌套
    public int arrayNesting(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int n = nums.length, ans = 0;
        for (int i = 0; i < n; i++) {
            if(nums[i] == -1) continue;
            int cnt = 0;
            while (nums[i] != -1){
                cnt++;
                int tmp = nums[i];
                nums[i] = -1;
                i = tmp;
            }
            ans = Math.max(cnt, ans);
        }
        return ans;
    }

    //https://leetcode.cn/problems/shift-2d-grid/ 二维网格迁移
    public List<List<Integer>> shiftGrid(int[][] grid, int k) {
        if(grid == null || grid.length == 0){
            return null;
        }
        int m = grid.length, n = grid[0].length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            List<Integer> row = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                row.add(0);
            }
            ans.add(row);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = (i * n + j + k) % (m * n);
                int row = idx / n, col = idx % n;
                ans.get(row).set(col, grid[i][j]);
            }
        }
        return ans;
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