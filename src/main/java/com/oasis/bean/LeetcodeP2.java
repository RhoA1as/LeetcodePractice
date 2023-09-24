package com.oasis.bean;

import org.junit.jupiter.api.Test;

import java.util.*;

public class LeetcodeP2 {

    //https://leetcode.cn/problems/min-max-game/ 极大极小游戏
    public int minMaxGame(int[] nums) {
        int n = nums.length;
        while (n != 1) {
            n /= 2;
            for (int i = 0; i < n; i++) {
                if ((i & 1) == 0) {
                    nums[i] = Math.min(nums[i * 2], nums[i * 2 + 1]);
                } else {
                    nums[i] = Math.max(nums[i * 2], nums[i * 2 + 1]);
                }
            }
        }
        return nums[0];
    }

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

    //https://leetcode.cn/problems/find-if-path-exists-in-graph/ 寻找图中是否存在路径
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        List<List<Integer>> children = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            children.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            children.get(edge[0]).add(edge[1]);
            children.get(edge[1]).add(edge[0]);
        }
        if (source == destination) return true;
        boolean[] visited =new boolean[n];
        Deque<Integer> deque = new ArrayDeque<>();
        deque.offer(source);
        visited[source] = true;
        while (!deque.isEmpty()) {
            int node = deque.poll();
            for (int child : children.get(node)) {
                if (!visited[child]) {
                    if (child == destination) return true;
                    deque.offer(child);
                    visited[child] = true;
                }
            }
        }
        return false;
    }

    //https://leetcode.cn/problems/count-number-of-homogenous-substrings/ 统计同构子字符串的数目
    public int countHomogenous(String s) {
        if (s == null || s.isEmpty()) return 0;
        final int MOD = 1000000007;
        char curr = s.charAt(0);
        long cnt = 1;
        long ans = 0;
        int n = s.length();
        for (int i = 1; i < n; i++) {
            char c = s.charAt(i);
            if (c == curr) {
                ++cnt;
            } else {
                ans += (cnt + 1) * cnt / 2;
                curr = c;
                cnt = 1;
            }
        }
        ans += (cnt + 1) * cnt / 2;
        return (int) (ans % MOD);
    }

    //https://leetcode.cn/problems/two-out-of-three/ 至少在两个数组中出现的值
    public List<Integer> twoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
        List<Integer> ans = new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums1) {
            map.put(i, 1);
        }
        for (int i : nums2) {
            map.put(i, map.getOrDefault(i, 0) | 2);
        }
        for (int i : nums3) {
            map.put(i, map.getOrDefault(i, 0) | 4);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if ((entry.getValue() & (entry.getValue() - 1)) != 0) {
                ans.add(entry.getKey());
            }
        }
        return ans;
    }

    // https://leetcode.cn/problems/lexicographically-smallest-string-after-applying-operations/ 执行操作后字典序最小的字符串
    public String findLexSmallestString(String s, int a, int b) {
        int n = s.length();
        String ans = s;
        s = s + s;
        boolean[] visited = new boolean[n];
        for (int i = 0; !visited[i]; i = (i + b) % n) {
            visited[i] = true;
            char[] array = s.substring(i, i + n).toCharArray();
            for (int j = 0; j < 10; j++) {
                for (int n1 = 1; n1 < n; n1 += 2) {
                    array[n1] = (char) ('0' + (array[n1] - '0' + a) % 10);
                }
                if ((b & 1) != 0) {
                    for (int k = 0; k < 10; k++) {
                        for (int l = 0; l < n; l += 2) {
                            array[l] = (char) ('0' + (array[l] - '0' + a) % 10);
                        }
                        String tmp = new String(array);
                        if (tmp.compareTo(ans) < 0) ans = tmp;
                    }
                } else {
                    String tmp = new String(array);
                    if (tmp.compareTo(ans) < 0) ans = tmp;
                }

            }
        }
        return ans;
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

    //https://leetcode.cn/problems/fruit-into-baskets/ 水果成篮
    public int totalFruit(int[] fruits) {
        int n = fruits.length;
        int[] cnt = new int[n];
        int l = 0, r = 0;
        for (int tot = 0; r < n; r++) {
            if(++cnt[fruits[r]] == 1) tot++;
            if(tot > 2){
                if(--cnt[fruits[l++]] == 0) tot--;
            }
        }
        return r - l;
    }

    // https://leetcode.cn/problems/shortest-path-in-binary-matrix/ 二进制矩阵中的最短路径
    public int shortestPathBinaryMatrix(int[][] grid) {
        if (grid == null || grid[0] == null || grid[0].length == 0) {
            return -1;
        }
        if (grid[0][0] == 1) {
            return -1;
        }
        int[][] d = {{0 , 1}, {0, -1}, {1, 0}, {-1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        int ans = 0, l = grid.length;
        boolean[][] visited = new boolean[l][l];
        Deque<int[]> deque = new ArrayDeque<>();
        deque.offer(new int[]{0, 0});
        visited[0][0] = true;
        while (!deque.isEmpty()) {
            int n = deque.size();
            ans++;
            for (int i = 0; i < n; i++) {
                int[] point = deque.poll();
                visited[point[0]][point[1]] = true;
                if (point[0] == l - 1 && point[1] == l - 1) return ans;
                for (int[] de : d) {
                    int x = point[0] + de[0], y = point[1] + de[1];
                    if (x < 0 || x >= l || y < 0 || y >= l) continue;
                    if (visited[x][y] || grid[x][y] != 0) continue;
                    deque.offer(new int[]{x, y});
                    visited[x][y] = true;
                }
            }
        }
        return -1;
    }

    // https://leetcode.cn/problems/delete-nodes-and-return-forest/ 删点成林
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        Set<Integer> set = new HashSet<>();
        for (int i : to_delete) {
            set.add(i);
        }
        List<TreeNode> ans = new ArrayList<>();
        dfsDelNodes(root, true, set, ans);
        return ans;
    }

    public TreeNode dfsDelNodes(TreeNode root, boolean isRoot, Set<Integer> set, List<TreeNode> ans) {
        if (root == null) return null;
        boolean isDelete = set.contains(root.val);
        root.left = dfsDelNodes(root.left, isDelete, set, ans);
        root.right = dfsDelNodes(root.right, isDelete, set, ans);
        if (isDelete) {
            return null;
        }
        if (isRoot) ans.add(root);
        return root;
    }

    // https://leetcode.cn/problems/remove-zero-sum-consecutive-nodes-from-linked-list/ 从链表中删去总和值为零的连续节点
    public ListNode removeZeroSumSublists(ListNode head) {
        ListNode tmp = new ListNode(0);
        tmp.next = head;
        Map<Integer, ListNode> map = new HashMap<>();
        int preSum = 0;
        ListNode flag = tmp;
        while (flag != null) {
            preSum += flag.val;
            map.put(preSum, flag);
            flag = flag.next;
        }
        preSum = 0;
        flag = tmp;
        while (flag != null) {
            preSum += flag.val;
            flag.next = map.get(preSum).next;
            flag = flag.next;
        }
        return tmp.next;
    }

    // https://leetcode.cn/problems/number-of-closed-islands/ 统计封闭岛屿的数目
    public int closedIsland(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    Deque<int[]> deque = new ArrayDeque<>();
                    deque.offer(new int[] {i, j});
                    grid[i][j] = 1;
                    boolean flag = true;
                    while (!deque.isEmpty()) {
                        int[] node = deque.poll();
                        int x = node[0], y = node[1];
                        if (x == 0 || y == 0 || x == m - 1 || y == n - 1) {
                            flag = false;
                        }
                        for (int[] dir : dirs) {
                            int nx = x + dir[0], ny = y + dir[1];
                            if (nx < 0 || ny < 0 || nx >= m || ny >= n || grid[nx][ny] == 1) {
                                continue;
                            }
                            grid[nx][ny] = 1;
                            deque.offer(new int[] {nx, ny});
                        }
                    }
                    if (flag) {
                        res++;
                    }
                }
            }
        }
        return res;
    }

    // https://leetcode.cn/problems/pond-sizes-lcci/ 水域大小
    public int[] pondSizes(int[][] land) {
        mLand = land;
        mLandRow = land.length;
        mLandCol = land[0].length;
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < mLandRow; i++) {
            for (int j = 0; j < mLandCol; j++) {
                if (land[i][j] == 0) {
                    land[i][j] = -1;
                    ans.add(dfsPondSize(i, j));
                }
            }
        }
        int size = ans.size();
        int[] res = new int[size];
        for (int i = 0; i < size; i++) {
            res[i] = ans.get(i);
        }
        Arrays.sort(res);
        return res;
    }

    int[][] mLand;
    int mLandRow;
    int mLandCol;

    public int dfsPondSize(int x, int y) {
        int res = 1;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int newX = x + i, newY = y + j;
                if (newX < 0 || newY < 0 || newX >= mLandRow || newY >= mLandCol || mLand[newX][newY] != 0) {
                    continue;
                }
                mLand[newX][newY] = -1;
                res += dfsPondSize(newX, newY);
            }
        }
        return res;
    }

    // https://leetcode.cn/problems/maximum-value-of-a-string-in-an-array/ 数组中字符串的最大值
    public int maximumValue(String[] strs) {
        int max = -1;
        for (String str : strs) {
            int s;
            try {
                s = Integer.parseInt(str);
            } catch (Exception e) {
                s = str.length();
            }
            max = Math.max(max, s);
        }
        return max;
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

    // https://leetcode.cn/problems/house-robber-iii/ 打家劫舍 III
    Map<TreeNode, Integer> s;
    Map<TreeNode, Integer> n;
    public int rob(TreeNode root) {
        s = new HashMap<>();
        n = new HashMap<>();
        dfsRob(root);
        return Math.max(s.getOrDefault(root, 0), n.getOrDefault(root, 0));
    }

    private void dfsRob(TreeNode root) {
        if (root == null) return;
        dfsRob(root.left);
        dfsRob(root.right);
        s.put(root, root.val + n.getOrDefault(root.left, 0) + n.getOrDefault(root.right, 0));
        n.put(root, Math.max(s.getOrDefault(root.left, 0), n.getOrDefault(root.left, 0))
                + Math.max(s.getOrDefault(root.right, 0), n.getOrDefault(root.right, 0)));
    }

    // https://leetcode.cn/problems/reorder-list/ 重排链表
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;
        ListNode midNode = getMidNode(head);
        ListNode tmp = midNode.next;
        midNode.next = null;
        ListNode f2 = reverseListNode(tmp);
        ListNode f1 = head;
        while (f1 != null && f2 != null) {
            ListNode t1 = f1.next, t2 = f2.next;
            f1.next = f2;
            f2.next = t1;
            f1 = t1;
            f2 = t2;
        }
    }

    private ListNode getMidNode(ListNode node) {
        ListNode fast = node, slow = node;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    private ListNode reverseListNode(ListNode node) {
        ListNode f = node, prev = null;
        while (f != null) {
            ListNode nxt = f.next;
            f.next = prev;
            prev = f;
            f = nxt;
        }
        return prev;
    }

    // https://leetcode.cn/problems/operations-on-tree/description/ 树上的操作
    class LockingTree {

        int[] mParent;

        int[] mLockUser;
        List<List<Integer>> mChildren;


        public LockingTree(int[] parent) {
            int n = parent.length;
            mParent = parent;
            mLockUser = new int[n];
            Arrays.fill(mLockUser, -1);
            mChildren = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                mChildren.add(new ArrayList<>());
            }
            for (int i = 0; i < n; i++) {
                int p = parent[i];
                if (p != -1) {
                    mChildren.get(p).add(i);
                }
            }
        }

        public boolean lock(int num, int user) {
            if (mLockUser[num] == -1) {
                mLockUser[num] = user;
                return true;
            }
            return false;
        }

        public boolean unlock(int num, int user) {
            if (mLockUser[num] == user) {
                mLockUser[num] = -1;
                return true;
            }
            return false;
        }

        public boolean upgrade(int num, int user) {
            boolean check = mLockUser[num] == -1 && checkParent(mParent[num]) && checkChildrenAndUnlock(num);
            if (check) {
                mLockUser[num] = user;
            }
            return check;
        }

        private boolean checkParent(int num) {
            if (num == -1) {
                return true;
            }
            if (mLockUser[num] != -1) {
                return false;
            }
            return checkParent(mParent[num]);
        }

        private boolean checkChildrenAndUnlock(int num) {
            boolean lock = mLockUser[num] != -1;
            mLockUser[num] = -1;
            for (int child : mChildren.get(num)) {
                lock |= checkChildrenAndUnlock(child);
            }
            return lock;
        }
    }

    // https://leetcode.cn/problems/lru-cache/ LRU 缓存
    class LRUCache {

        class Node {
            int val;
            int key;
            Node prev;
            Node next;

            public Node() {}
            public Node(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }

        int mCapacity;
        Node head;
        Node tail;
        Map<Integer, Node> map;

        public LRUCache(int capacity) {
            mCapacity = capacity;
            head = new Node();
            tail = new Node();
            head.next = tail;
            tail.prev = head;
            map = new HashMap<>();
        }

        public int get(int key) {
            Node node = map.get(key);
            if (node != null) {
                moveToFirst(node);
                return node.val;
            }
            return -1;
        }

        public void put(int key, int value) {
            Node node = map.get(key);
            if (node != null) {
                node.val = value;
                moveToFirst(node);
            } else {
                if (map.size() == mCapacity) {
                    Node last = removeLast();
                    map.remove(last.key);
                }
                node = new Node(key, value);
                map.put(key, node);
                addToFirst(node);
            }
        }

        private void removeNode(Node node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void addToFirst(Node node) {
            node.next = head.next;
            node.prev = head;
            head.next = node;
            node.next.prev = node;
        }

        private void moveToFirst(Node node) {
            removeNode(node);
            addToFirst(node);
        }

        private Node removeLast() {
            Node last = tail.prev;
            removeNode(last);
            return last;
        }
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

    // https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/ 根到叶路径上的不足节点
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        return checkSufficientSubset(root, 0, limit) ? root : null;
    }

    private boolean checkSufficientSubset(TreeNode root, int sum, int limit) {
        if (root == null) return false;
        if (root.left == null && root.right == null) return sum + root.val >= limit;
        boolean checkLeft = checkSufficientSubset(root.left, sum + root.val, limit);
        boolean checkRight = checkSufficientSubset(root.right, sum + root.val, limit);
        if (!checkLeft) root.left = null;
        if (!checkRight) root.right = null;
        return checkLeft || checkRight;
    }

    @Test
    public void testSplitArraySameAverage(){
        System.out.println(splitArraySameAverage(new int[]{1, 6, 1}));
    }

    //https://leetcode.cn/problems/split-array-with-same-average/ 数组的均值分割
    public boolean splitArraySameAverage(int[] nums) {
        if(nums == null || nums.length <= 1) return false;
        int sum = Arrays.stream(nums).sum();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            nums[i] = nums[i] * n - sum;
        }
        Set<Integer> set = new HashSet<>();
        int m = n / 2;
        for (int i = 1; i < (1 << m); i++) {
            int tot = 0;
            for (int j = 0; j < m; j++) {
                if((i & (1 << j)) != 0){
                    tot += nums[j];
                }
            }
            if(tot == 0) return true;
            set.add(tot);
        }
        int rSum = 0;
        for (int i = m; i < n; i++) {
            rSum += nums[i];
        }
        for (int i = 1; i < (1 << (n - m)); i++) {
            int tot = 0;
            for (int j = m; j < n; j++) {
                if((i & (1 << (j - m))) != 0){
                    tot += nums[j];
                }
            }
            if(tot == 0 || (tot != rSum && set.contains(-tot))){
                return true;
            }
        }
        return false;
    }

    //https://leetcode.cn/problems/maximum-frequency-stack/ 最大频率栈
    class FreqStack {

        private Map<Integer, Integer> freq;
        private Map<Integer, Deque<Integer>> grp;
        private int maxFreq;
        public FreqStack() {
            freq = new HashMap<>();
            grp = new HashMap<>();
        }

        public void push(int val) {
            int f = freq.getOrDefault(val, 0) + 1;
            grp.putIfAbsent(f, new ArrayDeque<>());
            grp.get(f).push(val);
            freq.put(val, f);
            maxFreq = Math.max(maxFreq, f);
        }

        public int pop() {
            int val = grp.get(maxFreq).pop();
            freq.put(val, maxFreq - 1);
            if(grp.get(maxFreq).isEmpty()){
                maxFreq--;
            }
            return val;
        }
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

    //https://leetcode.cn/problems/string-rotation-lcci/ 字符串轮转
    public boolean isFlipedString(String s1, String s2) {
        if(s1 == null || s2 == null) return false;
        return s1.length() == s2.length() && (s1 + s1).contains(s2);
    }

    //https://leetcode.cn/problems/zero-matrix-lcci/ 零矩阵
    public void setZeroes(int[][] matrix) {
        if (matrix == null) return;
        int m = matrix.length, n = matrix[0].length;
        if(m == 0 || n == 0) return;
        int[] rows = new int[m];
        int[] cols = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(matrix[i][j] == 0){
                    rows[i] = 1;
                    cols[j] = 1;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(rows[i] == 1 || cols[j] == 1) matrix[i][j] = 0;
            }
        }
    }

    //https://leetcode.cn/problems/swap-adjacent-in-lr-string/ 在LR字符串中交换相邻字符
    public boolean canTransform(String start, String end) {
        if(start == null || end == null || start.length() != end.length()){
            return false;
        }
        if(start.equals(end)) return true;
        int n = start.length(), a = 0, b = 0;
        while (a < n || b < n){
            while (a < n && start.charAt(a) == 'X') a++;
            while (b < n && end.charAt(b) == 'X') b++;
            if(a == n || b == n) break;
            if(start.charAt(a) != end.charAt(b)) return false;
            if(start.charAt(a) == 'L' && a < b) return false;
            if(start.charAt(a) == 'R' && a > b) return false;
            a++; b++;
        }
        return a == b;
    }

    //https://leetcode.cn/problems/maximum-ascending-subarray-sum/ 最大升序子数组和
    public int maxAscendingSum(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        int curr = nums[0], max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if(nums[i] > nums[i-1]) curr += nums[i];
            else curr = nums[i];
            max = Math.max(max, curr);
        }
        return max;
    }

    //https://leetcode.cn/problems/advantage-shuffle/ 优势洗牌
    public int[] advantageCount(int[] nums1, int[] nums2) {
        if(nums1 == null || nums2 == null) return null;
        int n = nums1.length;
        Integer[] idx1 = new Integer[n];
        Integer[] idx2 = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx1[i] = i;
            idx2[i] = i;
        }
        Arrays.sort(idx1, (x,y) -> nums1[x] - nums1[y]);
        Arrays.sort(idx2, (x,y) -> nums2[x] - nums2[y]);
        int left = 0, right = n - 1;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            if(nums1[idx1[i]] > nums2[idx2[left]]){
                ans[idx2[left++]] = nums1[idx1[i]];
            } else {
                ans[idx2[right--]] = nums1[idx1[i]];
            }
        }
        return ans;
    }

    //https://leetcode.cn/problems/k-th-symbol-in-grammar/ 第k个语法符号
    public int kthGrammar(int n, int k) {
        if(k == 1) return 0;
        if(k > (1 << (n - 2))){
            return 1 ^ kthGrammar(n - 1, k - (1 << (n - 2)));
        }
        return kthGrammar(n - 1, k);
    }

    //https://leetcode.cn/problems/shortest-bridge/ 最短的桥
    int gm;
    int gn;
    int[][] island;
    int[][] gd = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public int shortestBridge(int[][] grid) {
        if(grid == null) return -1;
        gm = grid.length;
        gn = grid[0].length;
        island = grid;
        Deque<int[]> deque = new ArrayDeque<>();
        int step = 0;
        for (int i = 0; i < gm; i++) {
            for (int j = 0; j < gn; j++) {
                if(grid[i][j] == 1){
                    dfsIsland(i, j, deque);
                    while (!deque.isEmpty()){
                        int size = deque.size();
                        for (int k = 0; k < size; k++) {
                            int[] node = deque.poll();
                            int row = node[0], col = node[1];
                            for (int[] dir : gd) {
                                int row0 = row + dir[0], col0 = col + dir[1];
                                if(row0 < 0 || row0 >= gm || col0 < 0 || col0 >= gn || grid[row0][col0] == -1){
                                    continue;
                                }
                                if(grid[row0][col0] == 1){
                                    return step;
                                } else if (grid[row0][col0] == 0) {
                                    grid[row0][col0] = -1;
                                    deque.offer(new int[]{row0, col0});
                                }
                            }
                        }
                        step++;
                    }
                }
            }
        }
        return step;
    }

    public void dfsIsland(int x, int y, Deque<int[]> deque){
        if(x < 0 || x >= gm || y < 0 || y >= gn || island[x][y] != 1){
            return;
        }
        island[x][y] = -1;
        deque.offer(new int[]{x, y});
        for (int[] dir : gd) {
            dfsIsland(x + dir[0], y + dir[1], deque);
        }
    }

    //https://leetcode.cn/problems/merge-in-between-linked-lists/ 合并两个链表
    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        ListNode la = list1;
        for (int i = 0; i < a - 1; i++) {
            la = la.next;
        }
        ListNode lb = la;
        for (int i = 0; i < b - a + 2; i++) {
            lb = lb.next;
        }
        la.next = list2;
        while (list2.next != null) list2 = list2.next;
        list2.next = lb;
        return list1;
    }

    //https://leetcode.cn/problems/decode-the-message/ 解密消息
    public String decodeMessage(String key, String message) {
        char[] map = new char[26];
        char a = 'a';
        for (int i = 0; i < key.length(); i++) {
            char c = key.charAt(i);
            if (c != ' ' && map[c - 'a'] == 0) {
                map[c - 'a'] = a++;
            }
        }
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < message.length(); i++) {
            char c = message.charAt(i);
            ans.append(c == ' ' ? c : map[c - 'a']);
        }
        return ans.toString();
    }

    //https://leetcode.cn/problems/evaluate-boolean-binary-tree/ 计算布尔二叉树的值
    public boolean evaluateTree(TreeNode root) {
        if (root.left == null) return root.val == 1;
        if (root.val == 2) return evaluateTree(root.left) || evaluateTree(root.right);
        return evaluateTree(root.left) && evaluateTree(root.right);
    }

    //https://leetcode.cn/problems/longest-well-performing-interval/ 表现良好的最长时间段
    public int longestWPI(int[] hours) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0, s = 0;
        for (int i = 0; i < hours.length; i++) {
            s += hours[i] > 8 ? 1 : -1;
            if (s > 0) {
                res = Math.max(res, i + 1);
            } else if (map.containsKey(s - 1)){
                res = Math.max(res, i - map.get(s - 1));
            }
            if (!map.containsKey(s)) {
                map.put(s, i);
            }
        }
        return res;
    }

    // https://leetcode.cn/problems/maximum-average-pass-ratio/ 最大平均通过率
    public double maxAverageRatio(int[][] classes, int extraStudents) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> {
            long v1 = (b[1] + 1L) * b[1] * (a[1] - a[0]);
            long v2 = (a[1] + 1L) * a[1] * (b[1] - b[0]);
            if (v1 == v2) return 0;
            return v1 < v2 ? 1 : -1;
        });
        for (int[] aClass : classes) {
            queue.offer(new int[]{aClass[0], aClass[1]});
        }
        for (int i = 0; i < extraStudents; i++) {
            int[] top = queue.poll();
            queue.offer(new int[]{top[0] + 1, top[1] + 1});
        }
        double res = 0;
        for (int i = 0; i < classes.length; i++) {
            int[] top = queue.poll();
            res += 1.0 * top[0] / top[1];
        }
        return res / classes.length;
    }
}

class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

//https://leetcode.cn/problems/online-stock-span/ 股票价格跨度
class StockSpanner {
    int idx;
    Deque<int[]> deque;
    public StockSpanner() {
        idx = -1;
        deque = new ArrayDeque<>();
        deque.offer(new int[]{idx, Integer.MAX_VALUE});
    }

    public int next(int price) {
        idx++;
        while (deque.peekLast()[1] <= price) {
            deque.pollLast();
        }
        int ans = idx - deque.peekLast()[0];
        deque.offer(new int[]{idx, price});
        return ans;
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