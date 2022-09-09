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

    //https://leetcode.cn/problems/binary-tree-pruning/ 二叉树减枝
    public TreeNode pruneTree(TreeNode root) {
        if(root == null) return null;
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if(root.left == null && root.right == null && root.val == 0) return null;
        return root;
    }

    //https://leetcode.cn/problems/set-intersection-size-at-least-two/ 设置交集大小至少为2
    public int intersectionSizeTwo(int[][] intervals) {
        if(intervals == null || intervals.length == 0) return 0;
        int ans = 2, n = intervals.length;
        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
        int pl = intervals[0][1] - 1, pr = intervals[0][1];
        for (int i = 1; i < n; i++) {
            if(pl >= intervals[i][0] && pr <= intervals[i][1]) continue;
            if(pr < intervals[i][0]){
                ans += 2;
                pl = intervals[i][1] -1;
                pr = intervals[i][1];
            } else if(pl < intervals[i][0]){
                ans++;
                if(pr == intervals[i][1]){
                    pl = intervals[i][1] - 1;
                } else {
                    pl = pr;
                    pr = intervals[i][1];
                }
            }
        }
        return ans;
    }

    //https://leetcode.cn/problems/complete-binary-tree-inserter/ 完全二叉树插入器
    class CBTInserter {
        List<TreeNode> mList;
        int mIdx;
        public CBTInserter(TreeNode root) {
            mList = new ArrayList<>();
            mList.add(root);
            int idx = 0;
            while (idx < mList.size()){
                TreeNode tmp = mList.get(idx++);
                if(tmp.left != null) mList.add(tmp.left);
                if(tmp.right != null) mList.add(tmp.right);
            }
        }

        public int insert(int val) {
            TreeNode curr = new TreeNode(val);
            while (mList.get(mIdx).left != null && mList.get(mIdx).right != null) mIdx++;
            TreeNode node = mList.get(mIdx);
            if(node.left == null){
                node.left = curr;
            } else if (node.right == null) {
                node.right = curr;
            }
            mList.add(curr);
            return node.val;
        }

        public TreeNode get_root() {
            return mList.get(0);
        }
    }

    @Test
    public void testFrac(){
        System.out.println(fractionAddition("-1/2+1/2+1/3"));
    }
    //https://leetcode.cn/problems/fraction-addition-and-subtraction/ 分数加减运算
    public String fractionAddition(String expression) {
        if(expression == null || expression.length() == 0) return expression;
        int idx = 0, n = expression.length();
        long a = 0, b = 1;
        while (idx < n){
            long c = 0, d = 0, sign = 1;
            char e = expression.charAt(idx);
            if(e == '+' || e == '-'){
                sign = e == '-' ? -1 : 1;
                idx++;
            }
            while (idx < n && Character.isDigit(expression.charAt(idx))){
                c = c * 10 + expression.charAt(idx++) - '0';
            }
            idx++;
            c *= sign;
            while (idx < n && Character.isDigit(expression.charAt(idx))){
                d = d * 10 + expression.charAt(idx++) - '0';
            }
            a = a * d + c * b;
            b *= d;
        }
        if(a == 0) return "0/1";
        long g = gcd(Math.abs(a), b);
        return new StringBuilder().append(a / g).append("/").append(b / g).toString();
    }

    private long gcd(long a, long b){
        long c = a % b;
        while (c != 0){
            a = b;
            b = c;
            c = a % b;
        }
        return b;
    }

    //https://leetcode.cn/problems/rank-transform-of-an-array/ 数组序号转换
    public int[] arrayRankTransform(int[] arr) {
        int n = arr.length;
        int[] tmp = new int[n];
        System.arraycopy(arr, 0, tmp, 0, n);
        Arrays.sort(tmp);
        Map<Integer,Integer> mapping = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if(!mapping.containsKey(tmp[i])){
                mapping.put(tmp[i], mapping.size() + 1);
            }
        }
        for (int i = 0; i < n; i++) {
            tmp[i] = mapping.get(arr[i]);
        }
        return tmp;
    }

    //https://leetcode.cn/problems/generate-a-string-with-characters-that-have-odd-counts/ 生成每种字符都是奇数个的字符串
    public String generateTheString(int n) {
        if((n & 1) != 0){
            return "a".repeat(n);
        }
        return new StringBuilder("a".repeat(n-1)).append("b").toString();
    }

    //https://leetcode.cn/problems/design-circular-queue/ 设计循环队列
    class MyCircularQueue {

        int mSize;
        int[] mQueue;

        int mHead;
        int mTail;

        public MyCircularQueue(int k) {
            mSize = k + 1;
            mQueue = new int[k+1];
        }

        public boolean enQueue(int value) {
            if(isFull()) return false;
            mQueue[mTail] = value;
            mTail = (mTail + 1) % mSize;
            return true;
        }

        public boolean deQueue() {
            if(isEmpty()) return false;
            mHead = (mHead + 1) % mSize;
            return true;
        }

        public int Front() {
            if(isEmpty()) return -1;
            return mQueue[mHead];
        }

        public int Rear() {
            if(isEmpty()) return -1;
            return mQueue[(mTail - 1 + mSize) % mSize];
        }

        public boolean isEmpty() {
            return mHead == mTail;
        }

        public boolean isFull() {
            return (mTail + 1) % mSize == mHead;
        }
    }

    //https://leetcode.cn/problems/orderly-queue/ 有序队列
    public String orderlyQueue(String s, int k) {
        if(k == 1){
            int i = 0, j = 1, k_ = 0, n = s.length();
            while (i < n && j < n && k_ < n){
                char a = s.charAt((i + k_) % n), b = s.charAt((j + k_) % n);
                if(a == b){
                    k_++;
                } else {
                    if(a > b){
                        i += k_ + 1;
                    } else {
                        j += k_ + 1;
                    }
                    if(i == j) i++;
                    k_ = 0;
                }
            }
            i = Math.min(i, j);
            return s.substring(i) + s.substring(0, i);
        } else {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            return new String(chars);
        }
    }

    //https://leetcode.cn/problems/add-one-row-to-tree/ 在二叉树中增加一行
    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        if(depth == 1){
            return new TreeNode(val, root, null);
        }
        List<TreeNode> level = new ArrayList<>();
        level.add(root);
        for (int i = 1; i < depth - 1; i++) {
            List<TreeNode> tmp = new ArrayList<>();
            int n = level.size();
            for (int j = 0; j < n; j++) {
                TreeNode curr = level.get(j);
                if(curr.left != null) tmp.add(curr.left);
                if(curr.right != null) tmp.add(curr.right);
            }
            level = tmp;
        }
        for (TreeNode node : level) {
            node.left = new TreeNode(val, node.left, null);
            node.right = new TreeNode(val, null, node.right);
        }
        return root;
    }

    @Test
    public void testExclusiveTime(){
        List<String> logs = new ArrayList<>();
        logs.add("0:start:0");
        logs.add("0:start:1");
        logs.add("0:start:2");
        logs.add("0:end:3");
        logs.add("0:end:4");
        logs.add("0:end:5");
        exclusiveTime(1, logs);
    }

    //https://leetcode.cn/problems/exclusive-time-of-functions/ 函数的独占时间
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] ans = new int[n];
        Deque<int[]> deque = new ArrayDeque<>();
        for (String log : logs) {
            String[] split = log.split(":");
            int idx = Integer.parseInt(split[0]);
            String type = split[1];
            int timestamp = Integer.parseInt(split[2]);
            if("start".equals(type)){
                if(!deque.isEmpty()){
                    ans[deque.peekLast()[0]] += timestamp - deque.peekLast()[1];
                }
                deque.offer(new int[]{idx, timestamp});
            } else {
                int[] t = deque.removeLast();
                ans[t[0]] += timestamp - t[1] + 1;
                if(!deque.isEmpty()){
                    deque.peekLast()[1] = timestamp + 1;
                }
            }
        }
        return ans;
    }

    //https://leetcode.cn/problems/solve-the-equation/ 求解方程
    public String solveEquation(String equation) {
        if(equation == null || equation.length() == 0){
            return "No solution";
        }
        char[] array = equation.toCharArray();
        int idx = 0, x = 0, num = 0, op = 1, n = array.length;
        while (idx < n){
            if(array[idx] == '+'){
                op = 1; idx++;
            } else if(array[idx] == '-'){
                op = -1; idx++;
            } else if (array[idx] == '=') {
                op = 1; num *= -1; x *= -1;
                idx++;
            } else {
                int j = idx;
                while (j < n && array[j] != '+' && array[j] != '-' && array[j] != '=') j++;
                if(array[j-1] == 'x'){
                    x += (j - 1 == idx ? 1 : Integer.parseInt(equation.substring(idx, j - 1))) * op;
                } else {
                    num += Integer.parseInt(equation.substring(idx, j)) * op;
                }
                idx = j;
            }
        }
        if(x == 0) return num == 0 ? "Infinite solutions" : "No solution";
        return "x=" + (num / -x);
    }

    //https://leetcode.cn/problems/reformat-the-string/ 重新格式化字符串
    public String reformat(String s) {
        if(s == null || s.length() == 0) return s;
        int n = s.length(), num = 0, cha = 0;
        StringBuilder a = new StringBuilder(), b = new StringBuilder();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if(Character.isDigit(c)){
                a.append(c);
                num++;
            } else {
                b.append(c);
                cha++;
            }
        }
        if(Math.abs(num - cha) > 1) return "";
        StringBuilder ans = new StringBuilder();
        boolean flag = num > cha;
        int len = num + cha;
        while (ans.length() < len){
            ans.append(flag ? a.charAt(--num) : b.charAt(--cha));
            flag = !flag;
        }
        return ans.toString();
    }

    //https://leetcode.cn/problems/design-circular-deque/ 设计循环双端队列
    class MyCircularDeque {

        private int[] mQueue;
        private int mSize;
        private int mTop;
        private int mTail;
        public MyCircularDeque(int k) {
            mQueue = new int[k+1];
            mSize = k+1;
        }

        public boolean insertFront(int value) {
            if(isFull()) return false;
            mTop = (mTop - 1 + mSize) % mSize;
            mQueue[mTop] = value;
            return true;
        }

        public boolean insertLast(int value) {
            if(isFull()) return false;
            mQueue[mTail] = value;
            mTail = (mTail + 1) % mSize;
            return true;
        }

        public boolean deleteFront() {
            if(isEmpty()) return false;
            mTop = (mTop + 1) % mSize;
            return true;
        }

        public boolean deleteLast() {
            if(isEmpty()) return false;
            mTail = (mTail - 1 + mSize) % mSize;
            return true;
        }

        public int getFront() {
            if(isEmpty()) return -1;
            return mQueue[mTop];
        }

        public int getRear() {
            if(isEmpty()) return -1;
            return mQueue[(mTail - 1 + mSize) % mSize];
        }

        public boolean isEmpty() {
            return mTop == mTail;
        }

        public boolean isFull() {
            return (mTail + 1) % mSize == mTop;
        }
    }

    //https://leetcode.cn/problems/design-an-ordered-stream/ 设计有序流
    class OrderedStream {

        private String[] mStream;
        private int ptr;
        public OrderedStream(int n) {
            mStream = new String[n + 1];
            ptr = 1;
        }

        public List<String> insert(int idKey, String value) {
            mStream[idKey] = value;
            List<String> ans = new ArrayList<>();
            while (ptr < mStream.length && mStream[ptr] != null){
                ans.add(mStream[ptr++]);
            }
            return ans;
        }
    }

    //https://leetcode.cn/problems/deepest-leaves-sum/ 层数最深叶子节点的和
    public int deepestLeavesSum(TreeNode root) {
        if(root == null) return 0;
        int ans = 0;
        Deque<TreeNode> deque = new ArrayDeque<>();
        deque.offer(root);
        while (!deque.isEmpty()){
            int size = deque.size();
            ans = 0;
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();
                ans += node.val;
                if(node.left != null) deque.offer(node.left);
                if(node.right != null) deque.offer(node.right);
            }
        }
        return ans;
    }

    //https://leetcode.cn/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/ 检查单词是否为句中其他单词的前缀
    public int isPrefixOfWord(String sentence, String searchWord) {
        if(sentence == null || searchWord == null){
            return -1;
        }
        String[] s = sentence.split(" ");
        for (int i = 0; i < s.length; i++) {
            if(s[i].startsWith(searchWord)) return i + 1;
        }
        return -1;
    }

    @Test
    public void testprintTree(){
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        printTree(root);
    }

    //https://leetcode.cn/problems/print-binary-tree/ 输出二叉树
    List<List<String>> mTree;
    int h;
    public List<List<String>> printTree(TreeNode root) {
        mTree = new ArrayList<>();
        if(root == null) return mTree;
        h = getTreeHeight(root) - 1;
        int m = h + 1, n = (1 << (h + 1)) - 1;
        for (int i = 0; i < m; i++) {
            List<String> level = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                level.add("");
            }
            mTree.add(level);
        }
        buildTree(root, 0, (n - 1) / 2);
        return mTree;
    }

    private void buildTree(TreeNode root, int x, int y){
        if(root == null) return;
        mTree.get(x).set(y, String.valueOf(root.val));
        buildTree(root.left, x + 1, y - (1 << (h - x - 1)));
        buildTree(root.right, x + 1, y + (1 << (h - x - 1)));
    }

    private int getTreeHeight(TreeNode root){
        if(root == null) return 0;
        return 1 + Math.max(getTreeHeight(root.left), getTreeHeight(root.right));
    }

    @Test
    public void testFindClosestElements(){
        findClosestElements(new int[]{1, 2, 3, 4, 5}, 4, 3);
    }

    //https://leetcode.cn/problems/find-k-closest-elements/ 找到 K 个最接近的元素
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int r = binarySearch(arr, x);
        int l = r - 1, n = arr.length;
        for (int i = 0; i < k; i++) {
            if (l < 0){
                r++;
            }else if(r >= n || Math.abs(arr[r] - x) >= Math.abs(arr[l] - x)){
                l--;
            }else {
                r++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = l + 1; i < r; i++) {
            ans.add(arr[i]);
        }
        return ans;
    }

    public int binarySearch(int[] arr, int x){
        int l = 0, r = arr.length - 1;
        while (l < r){
            int m = (l + r) >> 1;
            if(arr[m] < x){
                l = m + 1;
            } else {
                r = m;
            }
        }
        return l;
    }

    //https://leetcode.cn/problems/maximum-product-of-two-elements-in-an-array/ 数组中两元素的最大乘积
    public int maxProduct(int[] nums) {
        int a = Math.max(nums[0], nums[1]);
        int b = Math.min(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            if(nums[i] > a){
                b = a;
                a = nums[i];
            }else if(nums[i] > b){
                b = nums[i];
            }
        }
        return (a - 1) * (b - 1);
    }

    //https://leetcode.cn/problems/maximum-width-of-binary-tree/ 二叉树最大宽度
    Map<Integer, Integer> mMinIdx;
    public int widthOfBinaryTree(TreeNode root) {
        mMinIdx = new HashMap<>();
        return dfsMaxWidth(root, 0, 0);
    }

    public int dfsMaxWidth(TreeNode node, int depth, int idx){
        if(node == null) return 0;
        mMinIdx.putIfAbsent(depth, idx);
        return Math.max(idx - mMinIdx.get(depth) + 1,
                Math.max(dfsMaxWidth(node.left, depth + 1, 2 * idx),
                        dfsMaxWidth(node.right, depth + 1, 2 * idx + 1)));
    }

    //https://leetcode.cn/problems/maximum-binary-tree-ii/ 最大二叉树2
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode node = new TreeNode(val);
        TreeNode prev = null, curr = root;
        while (curr != null && curr.val > val){
            prev = curr;
            curr = curr.right;
        }
        node.left = curr;
        if(prev != null) {
            prev.right = node;
            return root;
        }
        return node;
    }

    //https://leetcode.cn/problems/validate-stack-sequences/ 验证栈序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        if(pushed == null || popped == null || pushed.length != popped.length){
            return false;
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0, j = 0; i < pushed.length; i++) {
            deque.offer(pushed[i]);
            while (!deque.isEmpty() && deque.peekLast() == popped[j]){
                deque.pollLast();
                j++;
            }
        }
        return deque.isEmpty();
    }

    //https://leetcode.cn/problems/longest-univalue-path/ 最长同值路径
    private int max;
    public int longestUnivaluePath(TreeNode root) {
        max = 0;
        dfsPath(root);
        return max;
    }

    public int dfsPath(TreeNode node){
        if(node == null) return 0;
        int l = dfsPath(node.left), r = dfsPath(node.right);
        int curr = 0, path = 0;
        if(node.left != null && node.left.val == node.val){
            path = l + 1;
            curr += l + 1;
        }
        if(node.right != null && node.right.val == node.val){
            path = Math.max(path, r + 1);
            curr += r + 1;
        }
        max = Math.max(max, curr);
        return path;
    }

    //https://leetcode.cn/problems/special-positions-in-a-binary-matrix/ 二进制矩阵中的特殊位置
    public int numSpecial(int[][] mat) {
        int m = mat.length, n = mat[0].length;
        int[] rowSum = new int[m];
        int[] colSum = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rowSum[i] += mat[i][j];
                colSum[j] += mat[i][j];
            }
        }
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(mat[i][j] == 1 && rowSum[i] == 1 && colSum[j] == 1) ans++;
            }
        }
        return ans;
    }

    //https://leetcode.cn/problems/find-duplicate-subtrees/ 寻找重复的子树
    Map<String, Integer> map;
    List<TreeNode> subTrees;
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        map = new HashMap<>();
        subTrees = new ArrayList<>();
        dfsDuplicateSubTrees(root);
        return subTrees;
    }

    private String dfsDuplicateSubTrees(TreeNode node){
        if(node == null) return "#";
        StringBuilder tree = new StringBuilder();
        String s = tree.append(node.val).append(",").append(dfsDuplicateSubTrees(node.left))
                .append(",").append(dfsDuplicateSubTrees(node.right)).toString();
        map.put(s, map.getOrDefault(s, 0) + 1);
        if(map.get(s) == 2){
            subTrees.add(node);
        }
        return s;
    }

    //https://leetcode.cn/problems/rearrange-spaces-between-words/ 重新排列单词间的空格
    public String reorderSpaces(String text) {
        int n = text.length();
        String[] split = text.trim().split("\\s+");
        int space = n;
        for (String s : split) {
            space -= s.length();
        }
        StringBuilder builder = new StringBuilder();
        int l = split.length;
        if(l == 1)
            return builder.append(split[0]).append(" ".repeat(space)).toString();
        String sp = " ".repeat(space / (l - 1));
        for (int i = 0; i < l; i++) {
            builder.append(split[i]);
            if(i != l - 1) builder.append(sp);
        }
        return builder.append(" ".repeat(space % (l - 1))).toString();
    }

    //https://leetcode.cn/problems/beautiful-arrangement-ii/ 优美的排列 II
    public int[] constructArray(int n, int k) {
        int[] ans = new int[n];
        int idx = 0;
        for (int i = 1; i < n - k; i++) {
            ans[idx++] = i;
        }
        int l = n - k, r = n;
        while (l <= r){
            ans[idx++] = l;
            if(l != r) ans[idx++] = r;
            l++;r--;
        }
        return ans;
    }

    //https://leetcode.cn/problems/crawler-log-folder/ 文件夹操作日志搜集器
    public int minOperations(String[] logs) {
        if(logs == null || logs.length == 0){
            return 0;
        }
        int depth = 0;
        for (String log : logs) {
            switch (log) {
                case "./":
                    break;
                case "../":
                    if(depth != 0) depth--;
                    break;
                default:
                    depth++;
            }
        }
        return depth;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
      this.val = val;
      this.left = left;
      this.right = right;
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