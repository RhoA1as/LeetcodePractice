package com.oasis;

import java.time.LocalDate;
import java.util.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

import org.junit.jupiter.api.Test;

/**
 * leetcode 刷题
 *
 */
public class LeetCode {

    @Test
    public void test(){
        int[] num = {3,5,2,1,6,4};
        wiggleSort(num);
        System.out.println(Arrays.toString(num));
    
    }

    //https://leetcode-cn.com/problems/construct-the-rectangle/ 构造矩阵
    public int[] constructRectangle(int area) {
        if(area <= 0){
            return null;
        }
        int w = (int)Math.sqrt(area);
        while(area % w != 0){
            w--;
        }
        return new int[]{area / w , w};
    }

    //https://leetcode-cn.com/problems/two-sum/ 两数之和
    public int[] twoSum(int[] nums, int target) {
        if(nums == null || nums.length == 0){
            return null;
        }
        Map<Integer,Integer> map = new HashMap<>();
        map.put(nums[0],0);
        for (int i = 1; i < nums.length; i++) {
            int val = target-nums[i];
            if(map.containsKey(val)){
                return new int[]{map.get(val),i};
            }else {
                map.put(nums[i], i);
            }
        }
        return null;
    }

    //https://leetcode-cn.com/problems/add-two-numbers/ 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1 == null || l2 == null){
            return null;
        }
        ListNode tempHead = new ListNode(0);
        ListNode temp = tempHead;
        int current, ext = 0;
        ListNode f1 = l1, f2 = l2;
        while(f1 != null || f2 != null){
            int val1 = 0, val2 = 0;
            if(f1 != null){
                val1 = f1.val;
                f1 = f1.next;
            }
            if(f2 != null){
                val2 = f2.val;
                f2 = f2.next;
            }
            current = val1 + val2 + ext;
            ext = current / 10;
            ListNode curr = new ListNode(current % 10);
            temp.next = curr;
            temp = temp.next;
        }
        if(ext != 0){
            ListNode curr = new ListNode(ext);
            temp.next = curr; 
        }
        return tempHead.next;
    }

    //https://leetcode-cn.com/problems/shopping-offers/ 大礼包
    Map<List<Integer>,Integer> min_price = new HashMap<>();
    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        if(price == null || special == null || needs == null){
            return 0;
        }
        int size = needs.size();
        List<List<Integer>> filter_special = new ArrayList<>(); 
        for (List<Integer> spe : special) {
            int total_count = 0, total_price = 0;
            for (int i = 0; i < size; i++) {
               total_count += spe.get(i);
               total_price += spe.get(i) * price.get(i);   
            }
            if(total_count > 0 && total_price > spe.get(size)){
                filter_special.add(spe);
            }
        }
        return dfs(price, filter_special, needs, size);
    }

    public int dfs(List<Integer> price, List<List<Integer>> filter_special, List<Integer> curr_needs, int n){
        if(!min_price.containsKey(curr_needs)){
            int minPrice = 0;
            for (int i = 0; i < n; i++) {
               minPrice += curr_needs.get(i) * price.get(i); 
            }
            for(List<Integer> spe : filter_special){
                List<Integer> nxtneeds = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if(spe.get(i) > curr_needs.get(i)){
                        break;
                    }
                    nxtneeds.add(curr_needs.get(i)-spe.get(i));
                }
                if(nxtneeds.size() == n){
                    minPrice = Math.min(minPrice,spe.get(n)+dfs(price, filter_special, nxtneeds, n)); 
                }
            }
            min_price.put(curr_needs, minPrice);
        }
        return min_price.get(curr_needs);
    }

    //https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/ 无重复字符最长字串
    public int lengthOfLongestSubstring(String s) {
        if(s == null || s.length() == 0){
            return 0;
        }
        char[] strs = s.toCharArray();
        int[] curr_length = new int[strs.length];
        Arrays.fill(curr_length, 1);
        Map<Character,Integer> memo = new HashMap<>();
        memo.put(strs[0], 0);
        int max_len = 1;
        for (int i = 1; i < strs.length; i++) {
            char curr = strs[i];
           if(!memo.containsKey(curr)){
                curr_length[i] += curr_length[i-1];
           }else{
                int pre = memo.get(curr);
                if((i - pre) > curr_length[i-1]){
                    curr_length[i] += curr_length[i-1]; 
                }else{
                    curr_length[i] = i - pre;
                }
           }
           memo.put(curr, i);
           max_len = Math.max(max_len, curr_length[i]);
        }
        return max_len;
    }

    //https://leetcode-cn.com/problems/longest-palindromic-substring/ 最长回文子串
    public String longestPalindrome(String s) {
        if(s == null || s.length() == 0){
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int l1 = expand(i, i, s);
            int l2 = expand(i, i+1, s);
            int curr = Math.max(l1, l2);
            if(curr > end - start + 1){
                start = i - (curr - 1) / 2;
                end = i + curr / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    public int expand(int start, int end, String s){
        while(start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)){
            start--;
            end++;
        }
        return end - start - 1;
    }

    //https://leetcode-cn.com/problems/search-a-2d-matrix-ii/ 搜索二维矩阵
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0){
            return false;
        }
        int row = 0, col = matrix[0].length - 1;
        while(row < matrix.length && col >= 0){
            if(matrix[row][col] == target){
                return true;
            }else if(matrix[row][col] < target){
                row++;
            }else{
                col--;
            }
        }
        return false;
    }

    //https://leetcode-cn.com/problems/next-greater-element-i/ 下一个更大元素1
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        if(nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0){
            return null;
        }
        Stack<Integer> stack = new Stack<>();
        Map<Integer,Integer> ans = new HashMap<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            int curr = nums2[i];
            while(!stack.isEmpty() && stack.peek() <= curr){
                stack.pop();
            }
            ans.put(curr,stack.isEmpty()? -1: stack.peek());
            stack.push(curr);
        }
        int[] res = new int[nums1.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = ans.get(nums1[i]);
        }
        return res;
    }

    @Test
    public void reorderedPowerOf2Test(){
        System.out.println(reorderedPowerOf2(1));
    } 
    //https://leetcode-cn.com/problems/reordered-power-of-2/ 重新排序得到2的幂
    public boolean reorderedPowerOf2(int n) {
        if(n <= 0){
            return false;
        }
        char[] nums = Integer.toString(n).toCharArray();
        Arrays.sort(nums);
        return backtrack(nums, 0, 0, new boolean[nums.length]);
    }

    public boolean backtrack(char[] nums, int num, int index, boolean[] isVisited){
        if(index == nums.length){
            return isPowerOfTwo(num);
        }
        for (int i = 0; i < nums.length; i++) {
            if((num == 0 && nums[i] == '0') || isVisited[i] || (i > 0 && !isVisited[i-1] && nums[i] == nums[i-1])){
                continue;
            }
            isVisited[i] = true;
            if(backtrack(nums, num * 10 + nums[i] - '0', index+1, isVisited)){
                return true;
            }
            isVisited[i] = false;
        }
        return false;
    }
    public boolean isPowerOfTwo(int n){
        return (n & (n-1)) == 0;
    }

    //https://leetcode-cn.com/problems/sum-of-beauty-in-the-array/ 数组美丽值求和
    public int sumOfBeauties(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int n = nums.length;
        int[] min = new int[n];
        int[] max = new int[n];
        int minVal = nums[n-1], maxVal = nums[0];
        for (int i = 1; i < n; i++) {
            max[i] = maxVal;
            maxVal = Math.max(maxVal, nums[i]);
        }
        for(int i = n-2; i >= 1; i--){
            min[i] = minVal;
            minVal = Math.min(minVal,nums[i]);
        }
        int res = 0;
        for (int i = 1; i < max.length - 1; i++) {
            if(nums[i] > max[i] && nums[i] < min[i]){
                res += 2;
            }else if(nums[i] > nums[i-1] && nums[i] < nums[i+1]){
                res++;
            }
        }
        return res;
    }

    //https://leetcode-cn.com/problems/single-number-iii/ 只出现一次的数字 III
    public int[] singleNumber(int[] nums) {
        if(nums == null || nums.length == 0){
            return null;
        }
        Map<Integer,Integer> count = new HashMap<>();
        int[] ans = new int[2];
        int idx = 0;
        for (int i : nums) {
            int currCount = count.getOrDefault(i, 0);
            count.put(i,currCount+1);
        }
        for (int i : count.keySet()) {
            if(count.get(i) == 1){
                ans[idx++] = i;
            }
            if(idx == 2){
                break;
            }
        }
        return ans;
    }

    public int[] singleNumber1(int[] nums) {
        if(nums == null || nums.length == 0){
            return null;
        }
        int val = 0;
        for (int i : nums) {
            val ^= i;
        }
        int flag = val == Integer.MIN_VALUE? val: val & (-val);
        int ans1 = 0, ans2 = 0;
        for (int i : nums) {
            if((i & flag) != 0){
                ans1 ^= i;
            }else{
                ans2 ^= i;
            }
        }
        return new int[]{ans1,ans2};
    }

    //https://leetcode-cn.com/problems/find-missing-observations/ 找出缺失的观测数据
    public int[] missingRolls(int[] rolls, int mean, int n) {
        if(rolls == null || rolls.length == 0 || n < 0){
            return new int[0];
        }
        int m = rolls.length;
        int currSum = Arrays.stream(rolls).sum();
        int missSum = (m + n) * mean - currSum;
        if(missSum < n || missSum > 6 * n){
            return new int[0];
        }
        int[] ans = new int[n];
        int val = missSum / n;
        int rest = missSum % n;
        for (int i = 0; i < n; i++) {
            ans[i] = val + (i < rest ? 1 : 0);
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/keyboard-row/ 键盘行
    public String[] findWords(String[] words) {
        if(words == null || words.length == 0){
            return null;
        }
        int[][] dir = new int[26][1];
        init(dir, "qwertyuiop", 0);
        init(dir, "asdfghjkl", 1);
        init(dir, "zxcvbnm", 2);
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if(check(dir,words[i])){
                ans.add(words[i]);
            }
        }
        return ans.toArray(new String[ans.size()]);
    }

    public void init(int[][] dir, String s, int idx){
        char[] line = s.toCharArray();
        for (int i = 0; i < line.length; i++) {
           dir[line[i]-'a'][0] = idx;
        }
    }

    public boolean check(int[][] dir,String s){
        char[] strs = s.toCharArray();
        int currIdx = dir[Character.toLowerCase(strs[0])-'a'][0];
        for (int i = 1; i < strs.length; i++) {
            if(dir[Character.toLowerCase(strs[i])-'a'][0] != currIdx){
                return false;
            }
        }
        return true;
    }

    //https://leetcode-cn.com/problems/distribute-candies/ 分糖果
    public int distributeCandies(int[] candyType) {
        if(candyType == null || candyType.length == 0){
            return 0;
        }
        int n = candyType.length;
        Arrays.sort(candyType);
        int type = 1;
        for (int i = 1; i < candyType.length; i++) {
            if(candyType[i] != candyType[i-1]){
                type++;
            }
        }
        return type > n/2? n/2: type;
    }

    //https://leetcode-cn.com/problems/delete-node-in-a-linked-list/ 删除链表中的节点
    public void deleteNode(ListNode node) {
        if(node == null){
            return;
        }
        ListNode flag = node, prev = null;
        while(flag.next != null){
            flag.val = flag.next.val;
            prev = flag;
            flag = flag.next;
        }
        prev.next = null;
    }

    @Test
    public void testBalancedString(){
        System.out.println(balancedString("QWER"));
    }

    //https://leetcode-cn.com/problems/replace-the-substring-for-balanced-string/ 替换子串得到平衡字符串
    public int balancedString(String s) {
        if(s == null || s.length() == 0){
            return 0;
        }
        int n = s.length(), l = 0, r = 0;
        int ans = n;
        int[] count = new int[26];
        for (int i = 0; i < n; i++) {
            count[s.charAt(i)- 'A']++;
        }
        while(r < n){
            count[s.charAt(r)- 'A']--;
            while(checkCount(count, n/4) && l < n){
                ans = Math.min(ans,r-l+1);
                count[s.charAt(l++)-'A']++;
            }
            r++;
        }
        return ans;
    }

    public boolean checkCount(int[] count, int average){
        return count['Q'-'A'] <= average && count['W'-'A'] <= average && 
            count['E'-'A'] <= average && count['R'-'A'] <= average;
    }

    //https://leetcode-cn.com/problems/valid-perfect-square/ 有效的完全平方数
    public boolean isPerfectSquare(int num) {
        if(num < 0){
            return false;
        }
        int l = 0, r = num;
        while(l <= r){
            int mid = l + (r-l)/2;
            long curr = (long)mid*mid;
            if(curr == num){
                return true;
            }else if(curr > num){
                r = mid - 1;
            }else {
                l = mid + 1;
            }
        }
        return false;
    }

    //https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/ 最长定差子序列
    public int longestSubsequence(int[] arr, int difference) {
       if(arr == null || arr.length == 0){
           return 0;
       }
       int ans = 0;
       Map<Integer,Integer> map = new HashMap<>();
       for (int i : arr) {
          map.put(i,map.getOrDefault(i-difference, 0)+1);
          ans = Math.max(ans, map.get(i));
       } 
       return ans;
    }

    //https://leetcode-cn.com/problems/missing-number/ 丢失的数字
    public int missingNumber(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int currSum = 0, n = nums.length;
        for (int i : nums) {
            currSum += i;
        }
        return (n+1)*n/2 - currSum;
    }

    //https://leetcode-cn.com/problems/range-addition-ii/ 范围求和 II
    public int maxCount(int m, int n, int[][] ops) {
        if(m <= 0 || n <= 0){
            return 0;
        }
        if(ops == null || ops.length == 0 || ops[0].length == 0){
            return m*n;
        }
        int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
        for (int[] is : ops) {
            minX = Math.min(minX, is[0]);
            minY = Math.min(minY, is[1]);
        }
        return minX*minY;
    }

    //https://leetcode-cn.com/problems/reverse-prefix-of-word/ 反转单词前缀
    public String reversePrefix(String word, char ch) {
        if(word == null || word.length() == 0){
            return word;
        }
        int idx = word.indexOf(ch, 0);
        if(idx == -1){
            return word;
        }
        char[] chars = word.toCharArray();
        int l = 0, r = idx;
        while(l < r){
            char temp = chars[l];
            chars[l++] = chars[r];
            chars[r--] = temp;

        }
        return String.valueOf(chars);
    }

    //https://leetcode-cn.com/problems/nZZqjQ/ 剑指 Offer II 073. 狒狒吃香蕉
    public int minEatingSpeed(int[] piles, int h) {
        if(piles == null || piles.length == 0){
            return 0;
        }
        if(h <= 0){
            return -1;
        }
        int l = 1, r = Arrays.stream(piles).max().getAsInt(), ans = Integer.MAX_VALUE;
        while(l <= r){
            int mid = l + (r-l)/2;
            if(getTime(mid, piles) <= h){
                r = mid - 1;
                ans = Math.min(mid,ans);
            }else {
                l = mid+1;
            }
        }
        return ans;
    }

    public int getTime(int speed, int[] piles){
        int hour = 0;
        for (int i = 0; i < piles.length; i++) {
            hour += (piles[i] + speed - 1)/speed;
        }
        return hour;
    }

    //https://leetcode-cn.com/problems/teemo-attacking/ 提莫攻击
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        if(timeSeries == null || timeSeries.length == 0){
            return 0;
        }
        int n = timeSeries.length, ans = 0, currDuration = timeSeries[0] + duration - 1;
        for (int i = 1; i < n; i++) {
            if(timeSeries[i] <= currDuration){
                ans += (timeSeries[i] - timeSeries[i-1]);
            }else {
                ans += duration;
            }
            currDuration = timeSeries[i] + duration - 1; 
        }
        return ans + duration;
    }

    //https://leetcode-cn.com/problems/JFETK5/ 剑指 Offer II 002. 二进制加法
    public String addBinary(String a, String b) {
        if(a == null || b == null || a.length() == 0 || b.length() == 0){
            return null;
        }
        int al = a.length(), bl = b.length(),exa = 0;
        int n = Math.max(al, bl);
        StringBuilder ans = new StringBuilder();
        for (int i = 1; i <= n; i++) {
            int an = al - i >= 0? a.charAt(al-i) - '0': 0;
            int bn = bl - i >= 0? b.charAt(bl-i) - '0': 0;
            int sum = an+bn+exa;
            ans.append(sum % 2);
            exa = sum / 2;
        }
        if(exa == 1){
            ans.append(exa);
        }
        return ans.reverse().toString();
    }

    //https://leetcode-cn.com/problems/guess-number-higher-or-lower-ii/ 猜数字大小 II
    public int getMoneyAmount(int n) {
       if(n <= 0){
           return 0;
       } 
       int[][] dp = new int[n+1][n+1];
       for (int i = n-1; i >= 1; i--) {
           for (int j = i+1; j <= n; j++) {
               dp[i][j] = j + dp[i][j-1];
               for (int k = i; k < j; k++) {
                   dp[i][j] = Math.min(dp[i][j],k+Math.max(dp[k+1][j], dp[i][k-1]));
               }
           }
       }
       return dp[1][n];
    }

    //https://leetcode-cn.com/problems/maximum-product-of-word-lengths/ 最大单词长度乘积
    public int maxProduct(String[] words) {
        if(words == null || words.length == 0){
            return 0;
        }
        Map<Integer,Integer> map = new HashMap<>();
        for (String word : words) {
            char[] chars = word.toCharArray();
            int val = 0;
            for (char c : chars) {
                val |= 1 << c - 'a';
            }
            if(chars.length > map.getOrDefault(val, 0)){
                map.put(val,chars.length);
            }
        }
        int ans = 0;
        for(int i : map.keySet()){
            for(int j : map.keySet()){
                if((i & j) == 0){
                    ans = Math.max(ans,map.get(i) * map.get(j));
                }
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/integer-replacement/ 整数替换
    Map<Integer,Integer> map = new HashMap<>();
    public int integerReplacement(int n) {
        if(n <= 0){
            return -1;
        }
        if(n == 1){
            return 0;
        }
        if(!map.containsKey(n)){
            if(n % 2 == 0){
                map.put(n, 1+integerReplacement(n/2));
            }else {
                map.put(n,2+Math.min(integerReplacement(n/2),integerReplacement(n/2+1)));
            }
        }
        return map.get(n);
    }

    //https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/ N叉树最大的深度
    public int maxDepth(Node root) {
        if(root == null){
            return 0;
        }
        int maxDepth = 0;
        for (Node child : root.children) {
            int path = maxDepth(child);
            maxDepth = Math.max(maxDepth,path);
        }
        return 1+maxDepth;
    }

    //https://leetcode-cn.com/problems/sum-swap-lcci/ 交换和
    public int[] findSwapValues(int[] array1, int[] array2) {
        if(array1 == null || array2 == null || array1.length == 0 || array2.length == 0){
            return new int[0];
        }
        int sum1 = 0, sum2 = 0;
        for (int i : array1) {
            sum1 += i;
        }
        Set<Integer> memo = new HashSet<>();
        for (int i : array2) {
            sum2 += i;
            memo.add(i);
        }
        int diff = sum1 - sum2;
        if(diff % 2 == 0){
            int val = diff / 2;
            for (int i : array1) {
                if(memo.contains(i - val)){
                    return new int[]{i,i-val};
                }
            }
        }
        return new int[0];
    }

    //https://leetcode-cn.com/problems/ways-to-make-a-fair-array/ 生成平衡数组的方案数
    public int waysToMakeFair(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int sumEven = 0, sumOdd = 0;
        for (int i = 0; i < nums.length; i++) {
            if(i % 2 == 0){
                sumEven += nums[i];
            }else {
                sumOdd += nums[i];
            }
        }
        int ans = 0;
        for (int i = nums.length - 1; i >= 0; i--) {
            if(i % 2 == 0){
                sumEven -= nums[i];
                if(sumEven == sumOdd){
                    ans++;
                }
                sumOdd += nums[i];
            } else {
                sumOdd -= nums[i];
                if(sumEven == sumOdd){
                    ans++;
                }
                sumEven += nums[i];
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/ 找到字符串中所有字母异位词
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> idxs = new ArrayList<>();
        if(s == null || p == null || s.length() < p.length()){
            return idxs;
        }
        int[] sCount = new int[26];
        int[] pCount = new int[26];
        for(int i = 0; i < p.length(); i++){
            sCount[s.charAt(i)-'a']++;
            pCount[p.charAt(i)-'a']++;
        }
        if(Arrays.equals(sCount, pCount)){
            idxs.add(0);
        }
        for (int i = 0, j = p.length(); j < s.length(); i++, j++) {
            sCount[s.charAt(i)-'a']--;
            sCount[s.charAt(j)-'a']++;
            if(Arrays.equals(sCount, pCount)){
                idxs.add(i+1);
            }
        }
        return idxs;
    }

    //https://leetcode-cn.com/problems/range-sum-of-bst/ 二叉搜索树的范围和
    public int rangeSumBST(TreeNode root, int low, int high) {
        if(root == null){
            return 0;
        }
        Deque<TreeNode> queue = new LinkedList<>();
        queue.addLast(root);
        int res = 0;
        while(!queue.isEmpty()){
            TreeNode curr = queue.removeFirst();
            if(curr == null){
                continue;
            }
            if(curr.val > high){
                queue.addLast(curr.left);
            }else if(curr.val < low){
                queue.addLast(curr.right);
            }else{
                res += curr.val;
                queue.addLast(curr.left);
                queue.addLast(curr.right);
            }
        }
        return res;
    }

    //https://leetcode-cn.com/problems/nth-digit/ 第N位数字
    public int findNthDigit(int n) {
        if(n <= 0){
            return -1;
        }
        int x = 1;
        long val;
        while((val = (long)(x * 9 * Math.pow(10,x-1))) < n){
            n -= val;
            x++;
        }
        int start = (int)Math.pow(10,x-1);
        int ext = n / x, idx = n % x;
        if(idx != 0){
            ext++;
        }
        int des = start + ext - 1;
        return idx == 0? des % 10: (des / (int)Math.pow(10,x-idx)) % 10;
    }

    //https://leetcode-cn.com/problems/consecutive-characters/ 连续字符
    public int maxPower(String s) {
        if(s == null || s.length() == 0){
            return 0;
        }
        char[] chars = s.toCharArray();
        int[] len = new int[chars.length];
        Arrays.fill(len, 1);
        int res = 1;
        for (int i = 1; i < len.length; i++) {
            if(chars[i] == chars[i-1]){
                len[i] = len[i-1] + 1;
                res = Math.max(res,len[i]);
            }
        }
        return res;
    }

    //https://leetcode-cn.com/problems/relative-ranks/ 相对名次
    public String[] findRelativeRanks(int[] score) {
        if(score == null || score.length == 0){
            return null;
        }
        int n = score.length;
        String[] desc = {"Gold Medal", "Silver Medal", "Bronze Medal"};
        int[][] temp = new int[n][2];
        for (int i = 0; i < n; i++) {
            temp[i][0] = score[i];
            temp[i][1] = i;
        }
        Arrays.sort(temp,(x,y) -> y[0] - x[0]);
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            if(i <= 2){
                ans[temp[i][1]] = desc[i];
            }else {
                ans[temp[i][1]] = Integer.toString(i+1);
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/ransom-note/ 赎金信
    public boolean canConstruct(String ransomNote, String magazine) {
        if(ransomNote == null || magazine == null || ransomNote.length() > magazine.length()){
            return false;
        }
        Map<Character,Integer> dire = new HashMap<>();
        char[] chars = magazine.toCharArray();
        char[] letter = ransomNote.toCharArray();
        for (char c : chars) {
           dire.put(c, dire.getOrDefault(c, 0)+1);
        }
        for (char c : letter) {
            if(!dire.containsKey(c) || dire.get(c) == 0){
                return false;
            }
            dire.put(c, dire.get(c)-1);
        }
        return true;
    }
    
    //https://leetcode-cn.com/problems/powx-n/ pow(x,n)
    public double myPow(double x, int n) {
        return n >= 0? quickMul(x,n): 1/quickMul(x,-n);
    }

    public double quickMul(double x, long n){
        if(n == 0){
            return 1.0;
        }
        double y = quickMul(x,n/2);
        return n % 2 == 0? y*y: y*y*x;
    }

    //https://leetcode-cn.com/problems/truncate-sentence/ 截断句子
    public String truncateSentence(String s, int k) {
        if(s == null || s.length() == 0){
            return s;
        }
        if(k < 0){
            return "";
        }
        int n = s.length();
        if(k >= n){
            return s;
        }
        String[] strings = s.split("\\s");
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < k; i++) {
            builder.append(strings[i]);
            if(i != k-1){
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    //https://leetcode-cn.com/problems/coloring-a-border/ 边界着色
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int m = grid.length, n = grid[0].length;
        boolean[][] isVisited = new boolean[m][n];
        List<int[]> border = new ArrayList<>();
        Deque<int[]> deque = new LinkedList<>();
        int[][] ways = {{0,1},{0,-1},{1,0},{-1,0}};
        int originColor = grid[row][col];
        deque.offer(new int[]{row,col});
        while(!deque.isEmpty()){
            int[] curr = deque.poll();
            int x = curr[0], y = curr[1];
            isVisited[x][y] = true;
            boolean isBoard = false;
            for (int i = 0; i < 4; i++) {
                int nextX = x + ways[i][0], nextY = y + ways[i][1];
                if(!(nextX >= 0 && nextX < m && nextY >= 0 && nextY < n && grid[nextX][nextY] == originColor)){
                    isBoard = true;
                }else if(!isVisited[nextX][nextY]){
                    deque.offer(new int[]{nextX,nextY});
                }
            }
            if(isBoard){
                border.add(curr);
            }
        }
        for (int[] ints : border) {
            grid[ints[0]][ints[1]] = color;
        }
        return grid;
    }

    @Test
    public void testCanArrange(){
        System.out.println(canArrange(new int[]{1,2,3,4}, 4));
    }

    //https://leetcode-cn.com/problems/check-if-array-pairs-are-divisible-by-k/ 检查数组对是否可以被 k 整除
    public boolean canArrange(int[] arr, int k) {
        if(arr == null || arr.length == 0){
            return false;
        }
        int n = arr.length;
        if(n % 2 != 0){
            return false;
        }
        int[] mod = new int[k];
        for (int i : arr) {
            mod[(i % k + k) % k]++;
        }
        for (int i = 1; i <= k / 2; i++) {
            if(mod[i] != mod[k-i]){
                return false;
            }
        }
        return mod[0] % 2 == 0;
    }

    //https://leetcode-cn.com/problems/valid-tic-tac-toe-state/ 有效的井字游戏
    public boolean validTicTacToe(String[] board) {
        if(board == null || board.length == 0){
            return false;
        }
        int countX = 0, countO = 0;
        for (String s : board) {
            for (char c : s.toCharArray()) {
                if(c == 'X'){
                    countX++;
                }else if(c == 'O'){
                    countO++;
                }
            }
        }
        if(countX != countO && countX - 1 != countO){
            return false;
        }
        if(isWin(board,'X') && countX - 1 != countO){
            return false;
        }
        if(isWin(board,'O') && countO != countX){
            return false;
        }
        return true;
    }

    public boolean isWin(String[] board, char c){
        for (int i = 0; i < 3; i++) {
            if(board[i].charAt(0) == c && board[i].charAt(1) == c && board[i].charAt(2) == c){
                return true;
            }
            if(board[0].charAt(i) == c && board[1].charAt(i) == c && board[2].charAt(i) == c){
                return true;
            }
        }
        if(board[0].charAt(0) == c && board[1].charAt(1) == c && board[2].charAt(2) == c){
            return true;
        }
        if(board[2].charAt(0) == c && board[1].charAt(1) == c && board[0].charAt(2) == c){
            return true;
        }
        return false;
    }

    //https://leetcode-cn.com/problems/loud-and-rich/ 喧闹和富有
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        if(richer == null || quiet == null){
            return new int[0];
        }
        int n = quiet.length;
        int[] counts = new int[n];
        int[] ans = new int[n];
        List<ArrayList<Integer>> ways = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ways.add(new ArrayList<>());
        }
        for (int[] ints : richer) {
            ways.get(ints[0]).add(ints[1]);
            counts[ints[1]]++;
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if(counts[i] == 0){
                deque.offer(i);
            }
            ans[i] = i;
        }
        while(!deque.isEmpty()){
            int x = deque.poll();
            for (Integer integer : ways.get(x)) {
                if(quiet[ans[x]] < quiet[ans[integer]]){
                    ans[integer] = ans[x];
                }
                if(--counts[integer] == 0){
                    deque.offer(integer);
                }
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/water-bottles/ 换酒问题
    public int numWaterBottles(int numBottles, int numExchange) {
        if(numBottles == 0){
            return 0;
        }
        int ans = numBottles;
        while(numBottles >= numExchange){
            int ext = numBottles / numExchange;
            ans += ext;
            numBottles %= numExchange;
            numBottles += ext;
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/battleships-in-a-board/ 甲板上的战舰
    public int countBattleships(char[][] board) {
        if(board == null){
            return 0;
        }
        int m = board.length, n = board[0].length, ans =0;
        if(m == 0 || n == 0){
            return 0;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(board[i][j] == 'X'){
                    if(i > 0 && board[i-1][j] == 'X'){
                        continue;
                    }
                    if(j > 0 && board[i][j-1] == 'X'){
                        continue;
                    }
                    ans++;
                }
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/find-the-town-judge/ 找到小镇的法官
    public int findJudge(int n, int[][] trust) {
        int[] in = new int[n];
        int[] out = new int[n];
        for (int[] ints : trust) {
            in[ints[1]-1]++;
            out[ints[0]-1]++;
        }
        for (int i = 0; i < n; i++) {
            if(in[i] == n-1 && out[i] == 0){
                return i+1;
            }
        }
        return -1;
    }

    //https://leetcode-cn.com/problems/heaters/ 供暖器
    public int findRadius(int[] houses, int[] heaters) {
        if(houses == null || heaters == null || houses.length == 0 || heaters.length == 0){
            return 0;
        }
        Arrays.sort(houses);
        Arrays.sort(heaters);
        int ans = 0, n = houses.length, m = heaters.length;
        for (int i = 0,j = 0; i < n; i++) {
            int currDistance = Math.abs(houses[i] - heaters[j]);
            while(j < m - 1 && currDistance >= Math.abs(houses[i] - heaters[j+1])){
                j++;
                currDistance = Math.abs(houses[i] - heaters[j]);
            }
            ans = Math.max(currDistance,ans);
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/day-of-the-year/ 一年中的第几天
    public int dayOfYear(String date) {
        if(date == null || date.length() == 0){
            return 0;
        }
        int ans = 0;
        int[] month = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        String[] split = date.split("-");
        int year = Integer.parseInt(split[0]);
        if(year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)){
            month[1]++;
        }
        for (int i = 0; i < Integer.parseInt(split[1]) - 1; i++) {
            ans += month[i];
        }
        return ans + Integer.parseInt(split[2]);
    }

    //https://leetcode-cn.com/problems/repeated-string-match/ 重复叠加字符串匹配
    public int repeatedStringMatch(String a, String b) {
        if(a == null || b == null){
            return -1;
        }
        if(b.equals("")){
            return 0;
        }
        int al = a.length(), bl = b.length();
        boolean[] chars = new boolean[26];
        for (int i = 0; i < al; i++) {
            chars[a.charAt(i) - 'a'] = true;
        }
        for (int i = 0; i < bl; i++) {
            if(!chars[b.charAt(i) - 'a']){
                return -1;
            }
        }
        int ans = bl / al;
        StringBuilder builder = new StringBuilder(a.repeat(ans));
        for (int i = 0; i <= 2; i++) {
            if(builder.indexOf(b) != -1){
                return ans;
            }
            builder.append(a);
            ans++;
        }
        return -1;
    }

    //https://leetcode-cn.com/problems/UlBDOe/solution/ 秋叶收藏集
    public int minimumOperations(String leaves) {
        if(leaves == null || leaves.length() == 0){
            return 0;
        }
        int n = leaves.length();
        int[][] states = new int[n][3];
        states[0][0] = leaves.charAt(0) == 'y'? 1: 0;
        states[0][1] = states[0][2] = states[1][2] = Integer.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            int isY = leaves.charAt(i) == 'y'? 1: 0;
            int isR = leaves.charAt(i) == 'r'? 1: 0;
            states[i][0] = states[i-1][0] + isY;
            states[i][1] = Math.min(states[i-1][1],states[i-1][0]) + isR;
            if(i >= 2){
                states[i][2] = Math.min(states[i-1][2],states[i-1][1]) + isY;
            }
        }
        return states[n-1][2];
    }

    //https://leetcode-cn.com/problems/occurrences-after-bigram/ BigRam分词
    public String[] findOcurrences(String text, String first, String second) {
        if(text == null || first == null || second == null || text.length() == 0){
            return new String[0];
        }
        String[] split = text.split(" ");
        ArrayList<String> ans = new ArrayList<>();
        int n = split.length;
        for (int i = 0; i < n - 2; i++) {
            if(split[i].equals(first) && split[i+1].equals(second)){
                ans.add(split[i+2]);
            }
        }
        return ans.toArray(new String[ans.size()]);
    }

    //https://leetcode-cn.com/problems/friends-of-appropriate-ages/ 适龄的朋友
    public int numFriendRequests(int[] ages) {
        if(ages == null || ages.length == 0){
            return 0;
        }
        Arrays.sort(ages);
        int l = 0, r = 0, ans = 0, n = ages.length;
        for (int age : ages) {
            if(age < 15){
                continue;
            }
            while(ages[l] <= 0.5 * age + 7){
                l++;
            }
            while(r + 1 < n && ages[r+1] <= age){
                r++;
            }
            ans += r - l;
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/construct-binary-search-tree-from-preorder-traversal/ 前序遍历二叉搜索树
    public TreeNode bstFromPreorder(int[] preorder) {
        if(preorder == null || preorder.length == 0){
            return null;
        }
        int[] inorder = Arrays.copyOfRange(preorder, 0, preorder.length);
        Arrays.sort(inorder);
        return buildTree(preorder,inorder);
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder == null || inorder == null){
            return null;
        }
        if(preorder.length == 0 || inorder.length == 0){
            return null;
        }
        if(preorder.length == 1){
            return new TreeNode(preorder[0]);
        }
        int rootIn = preorder[0];
        TreeNode root = new TreeNode(rootIn);
        int i = 0;
        for(;i < inorder.length;i++){
            if(inorder[i] == rootIn){
                break;
            }
        }
        root.left = buildTree(Arrays.copyOfRange(preorder,1,i+1),Arrays.copyOfRange(inorder,0,i));
        root.right = buildTree(Arrays.copyOfRange(preorder,i+1,preorder.length),Arrays.copyOfRange(inorder,i+1,inorder.length));
        return root;
    }

    public TreeNode dfsBuildTree(int[] preorder, int left, int right){
        if(left > right){
            return null;
        }
        TreeNode root = new TreeNode(preorder[left]);
        if(left == right){
            return root;
        }
        int l = left, r = right;
        while(l < r){
            int mid = l + (r - l + 1) / 2;
            if(preorder[mid] <= preorder[left]){
                l = mid;
            }else {
                r = mid - 1;
            }
        }
        root.left = dfsBuildTree(preorder, left + 1, l);
        root.right = dfsBuildTree(preorder, l + 1, right);
        return root;
    }

    //https://leetcode-cn.com/problems/elimination-game/ 消除游戏
    public int lastRemaining(int n) {
        int a1 = 1, count = n, k = 0, step = 1;
        while(count > 1){
            if(k % 2 == 0){
                a1 = a1 + step;
            }else {
                a1 = count % 2 == 0? a1: a1 + step;
            }
            k++;
            count = count >> 1;
            step = step << 1;
        }
        return a1;
    }

    //https://leetcode-cn.com/problems/day-of-the-week/ 一周中的第几天
    public String dayOfTheWeek(int day, int month, int year) {
        String[] week = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
        return week[LocalDate.of(year,month,day).getDayOfWeek().getValue() - 1];
    }

    //https://leetcode-cn.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/ 替换所有的句号
    public String modifyString(String s) {
        if(s == null || s.length() == 0){
            return s;
        }
        char[] chars = s.toCharArray();
        int n = chars.length;
        for (int i = 0; i < n; i++) {
            if(chars[i] == '?'){
                for (char j = 'a'; j < 'z'; j++) {
                    if((i > 0 && chars[i-1] == j) || (i < n-1 && chars[i+1] == j)){
                        continue;
                    }
                    chars[i] = j;
                    break;
                }
            }
        }
        return new String(chars);
    }

    //https://leetcode-cn.com/problems/simplify-path/ 简化路径
    public String simplifyPath(String path) {
        if(path == null || path.length() == 0){
            return path;
        }
        Deque<String> deque = new ArrayDeque<>();
        String[] split = path.split("/");
        for (String s : split) {
            if(s.equals("..")){
                if(!deque.isEmpty()){
                    deque.pollLast();
                }
            }else if(!(s.equals("") || s.equals("."))){
                deque.offerLast(s);
            }
        }
        StringBuilder ans = new StringBuilder();
        if(deque.isEmpty()){
            return "/";
        }
        int n = deque.size();
        for (int i = 0; i < n; i++) {
            ans.append("/").append(deque.poll());
        }
        return ans.toString();
    }

    //https://leetcode-cn.com/problems/slowest-key/ 按键持续时间最长的键
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        if(releaseTimes == null || releaseTimes.length == 0 || keysPressed == null || keysPressed.length() == 0){
            return ' ';
        }
        char[] chars = keysPressed.toCharArray();
        char ans = chars[0];
        int max = releaseTimes[0], n = chars.length;
        for (int i = 1; i < n; i++) {
            int time = releaseTimes[i] - releaseTimes[i-1];
            if(time > max){
                ans = chars[i];
                max = time;
            }else if(time == max){
                ans = (char) Math.max(ans,chars[i]);
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/check-subtree-lcci/检查子树
    public boolean checkSubTree(TreeNode t1, TreeNode t2) {
        if(t2 == null) return true;
        if(t1 == null) return false;
        if(t1.val == t2.val) return checkSubTree(t1.left, t2.left) && checkSubTree(t1.right, t2.right);
        return checkSubTree(t1.left, t2) || checkSubTree(t1.right, t2);
    }

    //https://leetcode-cn.com/problems/increasing-triplet-subsequence/ 递增的三元子序列
    public boolean increasingTriplet(int[] nums) {
        if(nums == null || nums.length < 3){
            return false;
        }
        int n = nums.length;
        int first = nums[0], second = Integer.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            if(nums[i] > second){
                return true;
            }else if(nums[i] > first){
                second = nums[i];
            }else {
                first = nums[i];
            }
        }
        return false;
    }

    //https://leetcode-cn.com/problems/largest-number-at-least-twice-of-others/ 至少是其他数字两倍的最大数
    public int dominantIndex(int[] nums) {
        if(nums == null || nums.length == 0){
            return -1;
        }
        int n = nums.length;
        if(n == 1){
            return 0;
        }
        int[][] index = new int[n][2];
        for (int i = 0; i < n; i++) {
            index[i][0] = i;
            index[i][1] = nums[i];
        }
        Arrays.sort(index,(x,y) -> y[1] - x[1]);
        return index[0][1] >= 2 * index[1][1]? index[0][0]: -1;
    }

    //https://leetcode-cn.com/problems/calculate-money-in-leetcode-bank/ 计算力扣银行的钱
    public int totalMoney(int n) {
        if(n <= 0){
            return 0;
        }
        int week = n / 7, ext = n % 7, ans = 0;
        int firstWeek = (1 + 7) * 7 / 2;
        ans += firstWeek * week + week * (week - 1) * 7 / 2;
        ans += (week + 1) * ext + (ext - 1) * ext / 2;
        return ans;
    }

    //https://leetcode-cn.com/problems/contains-duplicate-ii/ 存在重复元素 II
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if(nums == null || nums.length == 0){
            return false;
        }
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int n = nums[i];
            if(map.containsKey(n) && i - map.get(n) <= k){
                return true;
            }
            map.put(n, i);
        }
        return false;
    }

    //https://leetcode-cn.com/problems/stone-game-ix/ 石子游戏 IX
    public boolean stoneGameIX(int[] stones) {
        if(stones == null || stones.length == 0){
            return false;
        }
        int count0 = 0, count1 = 0, count2 = 0;
        for (int stone : stones) {
            int type = stone % 3;
            if (type == 0){
                count0++;
            }else if(type == 1){
                count1++;
            }else {
                count2++;
            }
        }
        if(count0 % 2 == 0){
            return count1 >= 1 && count2 >= 1;
        }
        return Math.abs(count1 - count2) > 2;
    }

    //https://leetcode-cn.com/problems/remove-palindromic-subsequences/ 删除回文子序列
    public int removePalindromeSub(String s) {
        if(s == null || s.length() == 0){
            return 0;
        }
        boolean flag = true;
        int l = 0, r = s.length() - 1;
        while (l < r){
            if(s.charAt(l++) != s.charAt(r--)){
                flag = false;
                break;
            }
        }
        return flag? 1: 2;
    }

    //https://leetcode-cn.com/problems/string-matching-in-an-array/ 数组中的字符串匹配
    public List<String> stringMatching(String[] words) {
        if(words == null || words.length == 0){
            return null;
        }
        Arrays.sort(words, Comparator.comparingInt(String::length));
        int n = words.length;
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                if(words[j].contains(words[i])){
                    ans.add(words[i]);
                    break;
                }
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/count-of-matches-in-tournament/ 比赛中的配对次数
    public int numberOfMatches(int n) {
        if(n <= 0){
            return 0;
        }
        int ans = 0;
        while(n > 1){
            int ext = n % 2;
            int matches = n / 2;
            n = ext + matches;
            ans += matches;
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/number-of-valid-words-in-a-sentence/ 句子中的有效单词数
    public int countValidWords(String sentence) {
        if(sentence == null || sentence.length() == 0){
            return 0;
        }
        int ans = 0;
        for (String s : sentence.split("\\s")) {
            if(isValid(s)) ans++;
        }
        return ans;
    }

    private boolean isValid(String s){
        if(s.length() == 0){
            return false;
        }
        boolean hasHyphens = false;
        int n = s.length();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if(Character.isDigit(c)){
                return false;
            }else if(c == '-'){
                if(hasHyphens){
                    return false;
                }
                if(i == 0 || i == n - 1 || !Character.isLowerCase(s.charAt(i-1)) || !Character.isLowerCase(s.charAt(i+1))){
                    return false;
                }
                hasHyphens = true;
            }else if((c == '!' || c == '.' || c == ',') && i != n-1){
                return false;
            }
        }
        return true;
    }
    //https://leetcode-cn.com/problems/longest-nice-substring/ 最长的美好子字符串
    private int maxPos;
    private int maxLen;

    public String longestNiceSubstring(String s) {
        if(s == null || s.length() == 0){
            return null;
        }
        int n = s.length(), type = 0;
        for (int i = 0; i < n; i++) {
            type |= 1 << (Character.toLowerCase(s.charAt(i)) - 'a');
        }
        type = Integer.bitCount(type);
        for (int i = 1; i <= type; i++) {
            checkNiceSubString(s, i);
        }
        return s.substring(maxPos, maxPos + maxLen);
    }

    public void checkNiceSubString(String s, int typeNum){
        int[] lower = new int[26];
        int[] upper = new int[26];
        int n = s.length(), count = 0;
        for (int l = 0, r = 0, total = 0; r < n; r++) {
            char c = s.charAt(r);
            int idx = Character.toLowerCase(c) - 'a';
            if(Character.isLowerCase(c)){
                lower[idx]++;
                if(lower[idx] == 1 && upper[idx] > 0){
                    count++;
                }
            }else {
                upper[idx]++;
                if(upper[idx] == 1 && lower[idx] > 0){
                    count++;
                }
            }
            total += lower[idx] + upper[idx] == 1? 1: 0;
            while (total > typeNum){
                char lc = s.charAt(l);
                int idxL = Character.toLowerCase(lc) - 'a';
                if(Character.isLowerCase(lc)){
                    lower[idxL]--;
                    if(lower[idxL] == 0 && upper[idxL] > 0){
                        count--;
                    }
                }else {
                    upper[idxL]--;
                    if(upper[idxL] == 0 && lower[idxL] > 0){
                        count--;
                    }
                }
                total -= lower[idxL] + upper[idxL] == 0? 1: 0;
                l++;
            }
            if(count == typeNum && r -l + 1 > maxLen){
                maxPos = l;
                maxLen = r -l + 1;
            }
        }
    }

    //https://leetcode-cn.com/problems/count-number-of-pairs-with-absolute-difference-k/ 差的绝对值为 K 的数对数目
    public int countKDifference(int[] nums, int k) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        Map<Integer,Integer> cnts = new HashMap<>();
        cnts.put(nums[0], 1);
        int ans = 0, n = nums.length;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            ans += cnts.getOrDefault(num - k, 0);
            ans += cnts.getOrDefault(num + k, 0);
            cnts.put(num, cnts.getOrDefault(num, 0) + 1);
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/simplified-fractions/ 最简分数
    public List<String> simplifiedFractions(int n) {
        if(n <= 1){
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int i = 1; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if(gcd(i, j) == 1){
                    ans.add(i + "/" + j);
                }
            }
        }
        return ans;
    }

    //欧几里得算法计算最大公约数
    public int gcd(int i, int j){
        return j == 0? i: gcd(j, i % j);
    }

    //https://leetcode-cn.com/problems/number-of-enclaves/ 飞地的数量
    int[][] dirs = {{1, 0}, {-1 , 0}, {0, 1}, {0, -1}};
    int[][] mGrid;
    boolean[][] isVisited;
    int m;
    int n;
    public int numEnclaves(int[][] grid) {
        if(grid == null || grid.length == 0 || grid[0].length == 0){
            return 0;
        }
        m = grid.length;
        n = grid[0].length;
        isVisited = new boolean[m][n];
        mGrid = grid;
        for (int i = 0; i < n; i++) {
            dfsEnclaves(0, i);
            dfsEnclaves(m -1, i);
        }
        for (int i = 0; i < m; i++) {
            dfsEnclaves(i, 0);
            dfsEnclaves(i, n - 1);
        }
        int ans = 0;
        for (int i = 1; i < m - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if(grid[i][j] == 1 && !isVisited[i][j]) ans++;
            }
        }
        return ans;
    }

    public void dfsEnclaves(int row, int col){
        if(row < 0 || row >= m || col < 0 || col >= n || isVisited[row][col]){
            return;
        }
        if(mGrid[row][col] != 1){
            return;
        }
        isVisited[row][col] = true;
        for (int[] dir : dirs) {
            dfsEnclaves(row + dir[0], col + dir[1]);
        }
    }

    //https://leetcode-cn.com/problems/maximum-number-of-balloons/ “气球” "balloon"的最大数量
    public int maxNumberOfBalloons(String text) {
        if(text == null || text.length() == 0){
            return 0;
        }
        int[] cnts = new int[5];
        int n = text.length();
        for (int i = 0; i < n; i++) {
            switch (text.charAt(i)){
                case 'a':
                    cnts[0]++;
                    break;
                case 'b':
                    cnts[1]++;
                    break;
                case 'l':
                    cnts[2]++;
                    break;
                case 'n':
                    cnts[3]++;
                    break;
                case 'o':
                    cnts[4]++;
                    break;
                default:
                    break;
            }
        }
        cnts[2] /= 2;
        cnts[4] /= 2;
        return Arrays.stream(cnts).min().getAsInt();
    }

    //https://leetcode-cn.com/problems/single-element-in-a-sorted-array/ 有序数组中的单一元素
    public int singleNonDuplicate(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l < r){
            int m = l + (r -l) / 2;
            if(nums[m] == nums[m ^ 1]){
                l = m + 1;
            }else {
                r = m;
            }
        }
        return nums[l];
    }

    //https://leetcode-cn.com/problems/lucky-numbers-in-a-matrix/ 矩阵中的幸运数
    public List<Integer> luckyNumbers (int[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0){
            return null;
        }
        int m = matrix.length, n = matrix[0].length;
        List<Integer> ans = new ArrayList<>();
        int[] rowMin = new int[m];
        int[] colMax = new int[n];
        for (int i = 0; i < m; i++) {
            rowMin[i] = Arrays.stream(matrix[i]).min().getAsInt();
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                colMax[i] = Math.max(colMax[i], matrix[j][i]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int val = matrix[i][j];
                if(val == rowMin[i]){
                    if(val == colMax[j]){
                        ans.add(val);
                    }
                    break;
                }
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/knight-probability-in-chessboard/ 骑士在棋盘上的概率
    public double knightProbability(int n, int k, int row, int column) {
        int[][] dirs = {{-2, -1}, {-2, 1}, {2, -1}, {2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}};
        double[][][] dp = new double[k+1][n][n];
        for (int i = 0; i <= k; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < n; l++) {
                    if(i == 0){
                        dp[i][j][l] = 1;
                    }else {
                        for (int[] dir : dirs) {
                            int x = j + dir[0], y = l + dir[1];
                            if(x >= 0 && x < n && y >=0 && y < n) {
                                dp[i][j][l] += dp[i - 1][x][y] / 8;
                            }
                        }
                    }
                }
            }
        }
        return dp[k][row][column];
    }

    //https://leetcode-cn.com/problems/pancake-sorting/ 煎饼排序
    public List<Integer> pancakeSort(int[] arr) {
        if(arr == null || arr.length == 0){
            return null;
        }
        List<Integer> ans = new ArrayList<>();
        int n = arr.length;
        int[] idxs = new int[n+1];
        for (int i = 0; i < n; i++) {
            idxs[arr[i]] = i;
        }
        for (int i = n; i >= 1; i--) {
            int idx = idxs[i];
            if(idx == i - 1){
                continue;
            }
            if(idx != 0){
                ans.add(idx + 1);
                reverse(arr, 0, idx, idxs);
            }
            ans.add(i);
            reverse(arr, 0, i - 1, idxs);
        }
        return ans;
    }

    public void reverse(int[] arr, int start, int end, int[] idxs){
        while(start < end){
            idxs[arr[start]] = end;
            idxs[arr[end]] = start;
            int temp = arr[start];
            arr[start++] = arr[end];
            arr[end--] = temp;
        }
    }

    //https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/ 1比特与2比特字符
    public boolean isOneBitCharacter(int[] bits) {
        if(bits == null || bits.length == 0){
            return false;
        }
        int idx = 0, n = bits.length;
        while(idx < n - 1){
            if(bits[idx] == 1){
                idx += 2;
            }else {
                idx++;
            }
        }
        return idx == n - 1;
    }

    //https://leetcode-cn.com/problems/push-dominoes/ 推多米诺
    public String pushDominoes(String dominoes) {
        if(dominoes == null || dominoes.length() == 0){
            return dominoes;
        }
        char[] chars = dominoes.toCharArray();
        int n = chars.length;
        int[] times = new int[n];
        Deque<int[]> deque = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            char c = chars[i];
            if(c == '.') continue;
            deque.offer(new int[]{i, 1, c == 'L'? -1: 1});
            times[i] = 1;
        }
        while (!deque.isEmpty()){
            int[] poll = deque.poll();
            int idx = poll[0], time = poll[1], force = poll[2];
            int new_idx = idx + force;
            if(new_idx < 0 || new_idx >=n || chars[idx] == '.') continue;
            if(times[new_idx] == 0){
                times[new_idx] = 1 + time;
                chars[new_idx] = chars[idx];
                deque.offer(new int[]{new_idx, time + 1, force});
            }else if(times[new_idx] == time + 1){
                chars[new_idx] = '.';
            }
        }
        return String.valueOf(chars);
    }

    //https://leetcode-cn.com/problems/making-file-names-unique/ 保证文件名唯一
    public String[] getFolderNames(String[] names) {
        if (names == null || names.length == 0) {
            return null;
        }
        Map<String, Integer> cnts = new HashMap<>();
        int n = names.length;
        String[] res = new String[n];
        for (int i = 0; i < n; i++) {
            String s = names[i];
            if (!cnts.containsKey(s)) {
                res[i] = s;
                cnts.put(s, 1);
            } else {
                int val = cnts.get(s);
                String new_name;
                while (cnts.containsKey((new_name = s + "(" + val + ")"))) {
                    val++;
                }
                res[i] = new_name;
                cnts.put(new_name, 1);
                cnts.put(s, val);
            }
        }
        return res;
    }

    //https://leetcode-cn.com/problems/reverse-only-letters/ 仅仅反转字母
    public String reverseOnlyLetters(String s) {
        if(s == null || s.length() == 0){
            return s;
        }
        char[] chars = s.toCharArray();
        int l = 0, r = chars.length - 1;
        while(l < r){
            while (l < r && !Character.isLetter(chars[l])){
                l++;
            }
            while (r > l && !Character.isLetter(chars[r])){
                r--;
            }
            if(l >= r){
                break;
            }
            char c = chars[l];
            chars[l++] = chars[r];
            chars[r--] = c;
        }
        return String.valueOf(chars);
    }

    //https://leetcode-cn.com/problems/where-will-the-ball-fall/ 球会落何处
    public int[] findBall(int[][] grid) {
        if(grid == null || grid.length == 0 || grid[0].length == 0){
            return new int[0];
        }
        int m = grid.length, n = grid[0].length;
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            int row = 0, col = i;
            while (true){
                int dir = grid[row][col];
                int nxt = col + dir;
                if(nxt < 0 || nxt == n || dir != grid[row][nxt]){
                    res[i] = -1;
                    break;
                }
                if(row == m - 1){
                    res[i] = nxt;
                    break;
                }
                row++;
                col = nxt;
            }
        }
        return res;
    }

    //https://leetcode-cn.com/problems/optimal-division/ 最优除法
    public String optimalDivision(int[] nums) {
        if(nums == null || nums.length == 0){
            return null;
        }
        int n = nums.length;
        if(n == 1){
            return String.valueOf(nums[0]);
        }
        if(n == 2){
            return nums[0] + "/" + nums[1];
        }
        StringBuilder builder = new StringBuilder();
        builder.append(nums[0]).append("/(").append(nums[1]);
        for (int i = 2; i < n; i++) {
            builder.append("/").append(nums[i]);
        }
        builder.append(")");
        return builder.toString();
    }

    //https://leetcode-cn.com/problems/count-vowel-substrings-of-a-string/ 统计字符串中的元音子字符串
    public int countVowelSubstrings(String word) {
        if(word == null || word.length() < 5){
            return 0;
        }
        int n = word.length(), ans = 0;
        char[] words = word.toCharArray();
        for (int i = 0; i < n; i++) {
            Set<Character> set = new HashSet<>();
            for (int j = i; j < n; j++) {
                if(!isVowel(words[j])){
                    break;
                }
                set.add(words[j]);
                if(set.size() == 5){
                    ans++;
                }
            }
        }
        return ans;
    }

    public boolean isVowel(char c){
        return c == 'a' || c == 'e' || c == 'i'
                || c == 'o' || c == 'u';
    }

    //https://leetcode-cn.com/problems/zigzag-conversion/ Z 字形变换
    public String convert(String s, int numRows) {
        if(s == null || s.length() == 0 || numRows == 1){
            return s;
        }
        List<StringBuilder> rows = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            rows.add(new StringBuilder());
        }
        int currRow = 0, changeRow = 1;
        for (char c : s.toCharArray()) {
            rows.get(currRow).append(c);
            if(currRow == 0 || currRow == numRows - 1){
                changeRow = currRow == 0? 1: -1;
            }
            currRow += changeRow;
        }
        StringBuilder ans = new StringBuilder();
        for (StringBuilder row : rows) {
            ans.append(row);
        }
        return ans.toString();
    }

    //https://leetcode-cn.com/problems/peaks-and-valleys-lcci/ 峰与谷
    public void wiggleSort(int[] nums) {
        if(nums == null || nums.length == 0){
            return;
        }
        int n = nums.length;
        int[] copy = Arrays.copyOfRange(nums, 0, n);
        Arrays.sort(copy);
        int l = 0, r = n-1, idx = 0;
        while (idx < n){
            nums[idx++] = idx % 2 == 0? copy[r--]: copy[l++];
        }
    }

    //https://leetcode-cn.com/problems/add-digits/ 各位相加
    public int addDigits_1(int num) {
        if(num < 10){
            return num;
        }
        while(num >= 10){
            int temp = 0;
            while(num > 0){
                temp += num % 10;
                num /= 10;
            }
            num = temp;
        }
        return num;
    }

    public int addDigits(int num) {
        return (num - 1) % 9 + 1;
    }

    //https://leetcode-cn.com/problems/find-good-days-to-rob-the-bank/ 适合打劫银行的日子
    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        if(security == null || security.length == 0){
            return null;
        }
        int n = security.length;
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 1; i < n; i++) {
            if(security[i] <= security[i-1]){
                left[i] = left[i-1] + 1;
            }
            if(security[n-i-1] <= security[n-i]){
                right[n-i-1] = right[n-i] + 1;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if(left[i] >= time && right[i] >= time){
                ans.add(i);
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/base-7/ 七进制数
    public String convertToBase7(int num) {
        if(num == 0){
            return "0";
        }
        boolean negative = num < 0;
        num = Math.abs(num);
        StringBuilder ans = new StringBuilder();
        while (num > 0){
            ans.append(num % 7);
            num /= 7;
        }
        if(negative) ans.append("-");
        return ans.reverse().toString();
    }

    //https://leetcode-cn.com/problems/plates-between-candles/ 蜡烛之间的盘子
    public int[] platesBetweenCandles(String s, int[][] queries) {
        if(s == null || s.length() == 0){
            return new int[0];
        }
        int n = s.length(), sum = 0, l = -1, r = -1;
        int[] preSum = new int[n];
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 0; i < n; i++) {
            if(s.charAt(i) == '*'){
                sum++;
            }
            preSum[i] = sum;
        }
        for (int i = 0; i < n; i++) {
            if(s.charAt(i) == '|') l = i;
            left[i] = l;
            if(s.charAt(n-i-1) == '|') r = n-i-1;
            right[n-i-1] = r;
        }
        int[] ans = new int[queries.length];
        int idx = 0;
        for (int[] query : queries) {
            int x = right[query[0]], y = left[query[1]];
            ans[idx++] = x == -1 || y == -1 || x >= y? 0: preSum[y] - preSum[x];
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/ 满足条件的子序列数目
    int mod = 1000000007;
    public int numSubseq(int[] nums, int target) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int ans = 0, n = nums.length;
        int[] pow = new int[n];
        pow[0] = 1;
        for (int i = 1; i < n; i++) {
            pow[i] = (pow[i-1] << 1) % mod;
        }
        Arrays.sort(nums);
        int l = 0, r = n-1;
        long res = 0;
        while(l <= r){
            int sum = nums[l] + nums[r];
            if(sum > target){
                r--;
            }else{
                res = (res + pow[r-l]) % mod;
                l++;
            }
        }
        return (int)res;
    }

    public long fastPow(long a, int n){
        long res = 1;
        while(n > 0){
            if((n & 1) != 0) res = (res * a) % mod;
            a = (a * a) % mod;
            n >>= 1;
        }
        return res;
    }

    //https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/ N 叉树的前序遍历
    public List<Integer> preorder(Node root) {
        if(root == null){
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        Stack<Node> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()){
            Node temp = stack.pop();
            ans.add(temp.val);
            List<Node> children = temp.children;
            for (int i = children.size() - 1; i >= 0; i--) {
                stack.push(children.get(i));
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/count-nodes-with-the-highest-score/ 统计最高分的节点数目
    long maxHeight;
    int cnts;
    int size;
    List<List<Integer>> children;

    public int countHighestScoreNodes(int[] parents) {
        if(parents == null || parents.length == 0){
            return 0;
        }
        int n = parents.length;
        size = n;
        children = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            children.add(new ArrayList<>());
        }
        for (int i = 1; i < n; i++) {
            children.get(parents[i]).add(i);
        }
        dfsCountHeights(0);
        return cnts;
    }

    public int dfsCountHeights(int node){
        long currAns = 1;
        int total = 1;
        List<Integer> list = children.get(node);
        for (Integer integer : list) {
            int totalChild = dfsCountHeights(integer);
            total += totalChild;
            currAns *= totalChild;
        }
        if(node != 0) currAns *= (size - total);
        if(currAns == maxHeight){
            cnts++;
        }else if(currAns > maxHeight){
            maxHeight = currAns;
            cnts = 1;
        }
        return total;
    }

    //https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/ N 叉树的后序遍历
    public List<Integer> postorder(Node root) {
        List<Integer> ans = new ArrayList<>();
        if(root == null) return ans;
        Stack<Node> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()){
            Node temp = stack.pop();
            ans.add(temp.val);
            for (Node child : temp.children) {
                stack.push(child);
            }
        }
        Collections.reverse(ans);
        return ans;
    }

    //https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/ 两个列表的最小索引总和
    public String[] findRestaurant(String[] list1, String[] list2) {
        if(list1 == null || list2 == null || list1.length == 0 || list2.length == 0){
            return new String[0];
        }
        Map<String,Integer> map = new HashMap<>();
        int n1 = list1.length, n2 = list2.length;
        for (int i = 0; i < n1; i++) {
            map.put(list1[i], i);
        }
        int idxSum = Integer.MAX_VALUE;
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < n2; i++) {
            String name = list2[i];
            if(!map.containsKey(name)){
                continue;
            }
            int currIdxSum = i + map.get(name);
            if(currIdxSum == idxSum){
                ans.add(name);
            }else if(currIdxSum < idxSum){
                ans.clear();
                ans.add(name);
                idxSum = currIdxSum;
            }
        }
        return ans.toArray(new String[ans.size()]);
    }

    //https://leetcode-cn.com/problems/count-number-of-maximum-bitwise-or-subsets/ 统计按位或能得到最大值的子集数目
    public int countMaxOrSubsets(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int maxVal = 0, cnt = 0, n = nums.length;
        for (int i = 1; i < (1 << n); i++) {
            int currVal = 0;
            for (int j = 0; j < n; j++) {
                if(((i >> j) & 1) != 0){
                    currVal |= nums[j];
                }
            }
            if(currVal > maxVal){
                maxVal = currVal;
                cnt = 1;
            }else if(currVal == maxVal){
                cnt++;
            }
        }
        return cnt;
    }

    //https://leetcode-cn.com/problems/longest-word-in-dictionary/ 词典中最长的单词
    public String longestWord(String[] words) {
        if(words == null || words.length == 0){
            return "";
        }
        Set<String> set = new HashSet<>();
        set.add("");
        String ans = "";
        Arrays.sort(words, (x,y) ->
                x.length() == y.length()? y.compareTo(x): x.length() - y.length());
        for (String word : words) {
            if(set.contains(word.substring(0, word.length() - 1))){
                ans = word;
                set.add(word);
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/construct-string-from-binary-tree/ 根据二叉树创建字符串
    public String tree2str(TreeNode root) {
        if(root == null){
            return "";
        }
        if(root.left == null && root.right == null){
            return Integer.toString(root.val);
        }
        if(root.right == null){
            return new StringBuilder().append(root.val).append("(").append(tree2str(root.left)).append(")").toString();
        }
        return new StringBuilder().append(root.val).append("(").append(tree2str(root.left)).append(")(").
                append(tree2str(root.right)).append(")").toString();
    }

    //https://leetcode-cn.com/problems/two-sum-iv-input-is-a-bst/ 两数之和 IV - 输入 BST
    Set<Integer> mValues = new HashSet<>();
    public boolean findTarget(TreeNode root, int k) {
        if(root == null) return false;
        if(mValues.contains(k - root.val)) return true;
        mValues.add(root.val);
        return findTarget(root.left, k) || findTarget(root.right, k);
    }

    @Test
    public void testWinnerOfGame(){
        System.out.println(winnerOfGame("ABBBBBBBAAA"));
    }

    //https://leetcode-cn.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/ 如果相邻两个颜色均相同则删除当前颜色
    public boolean winnerOfGame(String colors) {
        if(colors == null || colors.length() == 0){
            return false;
        }
        char c = colors.charAt(0);
        int cnt = 1, n = colors.length(), a = 0, b = 0;
        for (int i = 1; i < n; i++) {
            char curr = colors.charAt(i);
            if(curr != c){
                c = curr;
                cnt = 1;
            } else {
                if(++cnt >= 3){
                    if(curr == 'A'){
                        a++;
                    }else {
                        b++;
                    }
                }
            }
        }
        return a > b;
    }

    //https://leetcode-cn.com/problems/k-th-smallest-in-lexicographical-order/ 字典序的第K小数字
    public int findKthNumber(int n, int k) {
        if (k > n){
            return -1;
        }
        long prefix = 1, idx = 1;
        while (idx < k){
            long cnt = getCount(prefix, n);
            long curr = idx + cnt;
            if(curr <= k){
                idx = curr;
                prefix++;
            }else if(curr > k){
                idx++;
                prefix *= 10;
            }
        }
        return (int)prefix;
    }

    public long getCount(long prefix, long n){
         long cnt = 0, l = prefix, r = prefix + 1;
         while (l <= n){
             long realR = Math.min(n + 1, r);
             cnt += realR - l;
             l *= 10;
             r *= 10;
         }
         return cnt;
    }

    //https://leetcode-cn.com/problems/image-smoother/ 图片平滑器
    public int[][] imageSmoother(int[][] img) {
        if(img == null || img.length == 0 || img[0].length == 0){
            return new int[0][0];
        }
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        int m = img.length, n = img[0].length;
        int[][] ans = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int sum = img[i][j], cnt = 1;
                for (int[] dir : dirs) {
                    int row = i + dir[0], col = j + dir[1];
                    if(row < 0 || row >= m || col < 0 || col >= n){
                        continue;
                    }
                    sum += img[row][col];
                    cnt++;
                }
                ans[i][j] = sum / cnt;
            }
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/factorial-trailing-zeroes/ 阶乘后的零
    public int trailingZeroes(int n) {
        int ans = 0;
        while(n >= 5){
            ans += n / 5;
            n /= 5;
        }
        return ans;
    }

    //https://leetcode-cn.com/problems/baseball-game/  棒球比赛
    public int calPoints(String[] ops) {
        if(ops == null || ops.length == 0){
            return 0;
        }
        List<Integer> list = new ArrayList<>();
        int ans = 0;
        for (String op : ops) {
            int n = list.size();
            switch (op){
                case "C":
                    ans -= list.get(n-1);
                    list.remove(n-1);
                    break;
                case "D":
                    int d = list.get(n-1) * 2;
                    list.add(d);
                    ans += d;
                    break;
                case "+":
                    int p = list.get(n-1) + list.get(n-2);
                    list.add(p);
                    ans += p;
                    break;
                default:
                    int val = Integer.parseInt(op);
                    list.add(val);
                    ans += val;
            }
        }
        return ans;
    }
}
//https://leetcode-cn.com/problems/all-oone-data-structure/ 全 O(1) 的数据结构
class AllOne {
    Map<String,LfuNode> map;
    LfuNode root;

    public AllOne() {
        map = new HashMap<>();
        root = new LfuNode();
        root.pre = root;
        root.nxt = root;
    }

    public void inc(String key) {
        if(!map.containsKey(key)){
            if(root.nxt != root && root.nxt.cnt == 1){
                root.nxt.list.add(key);
                map.put(key, root.nxt);
            }else {
                LfuNode currNode = new LfuNode(key, 1);
                root.insert(currNode);
                map.put(key,currNode);
            }
        }else {
            LfuNode lfuNode = map.get(key);
            int cnt = lfuNode.cnt;
            Set<String> list = lfuNode.list;
            list.remove(key);
            if(lfuNode.nxt != root && lfuNode.nxt.cnt == cnt + 1){
                lfuNode.nxt.list.add(key);
                map.put(key, lfuNode.nxt);
            }else {
                LfuNode currNode = new LfuNode(key, cnt + 1);
                lfuNode.insert(currNode);
                map.put(key,currNode);
            }
            if(list.isEmpty()){
                lfuNode.remove();
            }
        }
    }

    public void dec(String key) {
        if(!map.containsKey(key)){
            return;
        }
        LfuNode lfuNode = map.get(key);
        Set<String> list = lfuNode.list;
        list.remove(key);
        int cnt = lfuNode.cnt;
        if(cnt - 1 != 0){
            if(lfuNode.pre != root && lfuNode.pre.cnt == cnt - 1){
                lfuNode.pre.list.add(key);
                map.put(key,lfuNode.pre);
            }else {
                LfuNode currNode = new LfuNode(key, cnt - 1);
                lfuNode.pre.insert(currNode);
                map.put(key,currNode);
            }
        }else {
            map.remove(key);
        }
        if(list.isEmpty()){
            lfuNode.remove();
        }
    }

    public String getMaxKey() {
        return root.pre == root? "": root.pre.list.iterator().next();
    }

    public String getMinKey() {
        return root.nxt == root? "": root.nxt.list.iterator().next();
    }

    class LfuNode{
        Set<String> list;
        int cnt;
        LfuNode pre;
        LfuNode nxt;

        public LfuNode() {
        }

        public LfuNode(String s, int cnt){
            list = new HashSet<>();
            list.add(s);
            this.cnt = cnt;
        }

        public void remove(){
            pre.nxt = nxt;
            nxt.pre = pre;
        }

        //在当前节点后插入
        public void insert(LfuNode node){
            node.nxt = nxt;
            nxt.pre = node;
            nxt = node;
            node.pre = this;
        }
    }
}

//https://leetcode-cn.com/problems/shuffle-an-array/ 打乱数组
class Solution {

    int[] nums;
    int[] origin;

    public Solution(int[] nums) {
        this.nums = nums;
        origin = new int[nums.length];
        System.arraycopy(nums, 0, origin, 0, nums.length);
    }
    
    public int[] reset() {
        System.arraycopy(origin, 0, nums, 0, nums.length);
        return nums;
    }
    
    public int[] shuffle() {
        Random random = new Random();
        for (int i = 0; i < nums.length; i++) {
            int idx = i + random.nextInt(nums.length - i);
            int temp = nums[i];
            nums[i] = nums[idx];
            nums[idx] = temp; 
        }
        return nums;
    }
}
class Solution2{
    Map<Integer,Integer> idxsMapping = new HashMap<>();
    int m, n, count;
    Random rand = new Random();
    public Solution2(int m, int n) {
        this.m = m;
        this.n = n;
        count = m * n;
    }
    
    public int[] flip() {
        int x = rand.nextInt(count--);
        int idx = idxsMapping.getOrDefault(x, x);
        idxsMapping.put(x,idxsMapping.getOrDefault(count, count));
        return new int[]{idx/n,idx%n};
    }
    
    public void reset() {
        count = m * n;
        idxsMapping.clear();
    }
}

class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};

class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}
//https://leetcode-cn.com/problems/map-sum-pairs/ 键值映射
class MapSum {

    class TriNode{
        int val;
        TriNode[] next = new TriNode[26];
    }

    TriNode root;
    Map<String,Integer> map;
    public MapSum() {
        root = new TriNode();
        map = new HashMap<>();
    }
    
    public void insert(String key, int val) {
        int change = val - map.getOrDefault(key, 0);
        char[] chars = key.toCharArray();
        TriNode curr = root;
        for (char c : chars) {
           if(curr.next[c-'a'] == null){
               curr.next[c-'a'] = new TriNode();
           } 
           curr = curr.next[c-'a'];
           curr.val += change;
        }
        map.put(key, val);
    }
    
    public int sum(String prefix) {
        char[] chars = prefix.toCharArray();
        TriNode curr = root;
        for (char c : chars) {
            if(curr.next[c-'a'] == null){
                return 0;
            }
            curr = curr.next[c-'a'];
        }
        return curr.val;
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

//https://leetcode-cn.com/problems/online-election/ 在线选举
class TopVotedCandidate {
    private Map<Integer,Integer> count;
    private int[] last;
    private int[] times;
    public TopVotedCandidate(int[] persons, int[] times) {
        int n = persons.length;
        count = new HashMap<>();
        last = new int[n];
        this.times = times;
        int top = 0;
        for (int i = 0; i < persons.length; i++) {
            int p = persons[i];
            count.put(p,count.getOrDefault(p,0) + 1);
            if (i == 0){
                continue;
            }
            if(count.get(p) >= count.get(top)){
                last[i] = p;
                top = p;
            }else {
                last[i] = top;
            }
        }
    }

    public int q(int t) {
        int l = 0, r = times.length - 1;
        while(l < r){
            int m = l + (r - l + 1) / 2;
            if(times[m] <= t){
                l = m;
            }else {
                r = m - 1;
            }
        }
        return last[l];
    }
}

//https://leetcode-cn.com/problems/stock-price-fluctuation/ 股票价格波动
class StockPrice {
    Map<Integer,Integer> map;
    TreeMap<Integer,Integer> treeMap;
    int currentTimestamp;
    public StockPrice() {
        map = new HashMap<>();
        treeMap = new TreeMap<>();
        currentTimestamp = 0;
    }

    public void update(int timestamp, int price) {
        currentTimestamp = Math.max(currentTimestamp, timestamp);
        if(map.containsKey(timestamp)){
            int prevPrice = map.get(timestamp);
            int count = treeMap.get(prevPrice);
            if(count == 1){
                treeMap.remove(prevPrice);
            }else {
                treeMap.put(prevPrice, count - 1);
            }
        }
        treeMap.put(price, treeMap.getOrDefault(price, 0) + 1);
        map.put(timestamp, price);
    }

    public int current() {
        return map.getOrDefault(currentTimestamp, 0);
    }

    public int maximum() {
        return treeMap.isEmpty()? 0: treeMap.lastKey();
    }

    public int minimum() {
        return treeMap.isEmpty()? 0: treeMap.firstKey();
    }
}

//https://leetcode-cn.com/problems/fizz-buzz-multithreaded/ 交替打印字符串
class FizzBuzz {
    private int n;
    private int i = 1;
    private ReentrantLock lock = new ReentrantLock();
    private Condition condition = lock.newCondition();
    public FizzBuzz(int n) {
        this.n = n;
    }

    // printFizz.run() outputs "fizz".
    public void fizz(Runnable printFizz) throws InterruptedException {
        while (i <= n){
            lock.lock();
            if(i % 3 == 0 && i % 5 != 0){
                printFizz.run();
                i++;
                condition.signalAll();
            }else {
                condition.await();
            }
            lock.unlock();
        }
    }

    // printBuzz.run() outputs "buzz".
    public void buzz(Runnable printBuzz) throws InterruptedException {
        while (i <= n){
            lock.lock();
            if(i % 3 != 0 && i % 5 == 0){
                printBuzz.run();
                i++;
                condition.signalAll();
            }else {
                condition.await();
            }
            lock.unlock();
        }
    }

    // printFizzBuzz.run() outputs "fizzbuzz".
    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
        while (i <= n){
            lock.lock();
            if(i % 3 == 0 && i % 5 == 0){
                printFizzBuzz.run();
                i++;
                condition.signalAll();
            }else {
                condition.await();
            }
            lock.unlock();
        }
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void number(IntConsumer printNumber) throws InterruptedException {
        while (i <= n){
            lock.lock();
            if(i % 3 != 0 && i % 5 != 0){
                printNumber.accept(i);
                i++;
                condition.signalAll();
            }else {
                condition.await();
            }
            lock.unlock();
        }
    }

    //https://leetcode-cn.com/problems/simple-bank-system/ 简易银行系统
    class Bank {

        private long[] mBalance;
        private int cnt;

        public Bank(long[] balance) {
            mBalance = balance;
            cnt = balance.length;
        }

        public boolean transfer(int account1, int account2, long money) {
            if(isIllegalAccount(account1) || isIllegalAccount(account2)
                    || mBalance[account1 - 1] < money){
                return false;
            }
            mBalance[account1 - 1] -= money;
            mBalance[account2 - 1] += money;
            return true;
        }

        public boolean deposit(int account, long money) {
            if(isIllegalAccount(account)) return false;
            mBalance[account - 1] += money;
            return true;
        }

        public boolean withdraw(int account, long money) {
            if(isIllegalAccount(account) || mBalance[account - 1] < money) return false;
            mBalance[account - 1] -= money;
            return true;
        }

        public boolean isIllegalAccount(int account){
            return account <= 0 || account > cnt;
        }
    }
}
