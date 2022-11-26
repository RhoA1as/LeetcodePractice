package com.oasis

import kotlin.math.*

//https://leetcode.cn/problems/largest-substring-between-two-equal-characters/ 两个相同字符之间的最长子字符串
fun maxLengthBetweenEqualCharacters(s: String): Int {
    if(s.isNullOrEmpty()) return -1
    val map = mutableMapOf<Char, Int>()
    var max = -1
    for ((idx, c) in s.withIndex()){
        if(c in map) max = max(max, idx - map.getValue(c) - 1)
        else map[c] = idx
    }
    return max
}

//https://leetcode.cn/problems/partition-array-into-disjoint-intervals/ 分割数组
fun partitionDisjoint(nums: IntArray): Int {
    if(nums.isEmpty()) return -1
    var maxVal = nums[0]
    var leftMax = nums[0]
    var position = 0
    val n = nums.size
    for (i in 1 until n){
        maxVal = maxVal.coerceAtLeast(nums[i])
        if(nums[i] < leftMax){
            position = i
            leftMax = maxVal
        }
    }
    return position + 1
}

//https://leetcode.cn/problems/reach-a-number/ 到达终点数字
fun reachNumber(target: Int): Int {
    var t = abs(target)
    var k = 0
    while (t > 0){
        k++
        t -= k
    }
    return if (t and 1 == 0) k else k + 1 + (k and 1)
}

//https://leetcode.cn/problems/largest-plus-sign/ 最大加号标志
fun orderOfLargestPlusSign(n: Int, mines: Array<IntArray>): Int {
    val dp = Array(n){ IntArray(n){n} }
    val set = mutableSetOf<Int>().apply {
        for((row, col) in mines)
            add(row * n + col)
    }
    var ans = 0
    var cnt: Int
    for (i in 0 until n){
        cnt = 0
        for(j in 0 until n){
            cnt = if(set.contains(i * n + j)) 0 else cnt + 1
            dp[i][j] = min(dp[i][j], cnt)
        }
        cnt = 0
        for(j in n - 1 downTo 0){
            cnt = if(set.contains(i * n + j)) 0 else cnt + 1
            dp[i][j] = min(dp[i][j], cnt)
        }
    }
    for (i in 0 until n){
        cnt = 0
        for(j in 0 until n){
            cnt = if(set.contains(j * n + i)) 0 else cnt + 1
            dp[j][i] = min(dp[j][i], cnt)
        }
        cnt = 0
        for(j in n - 1 downTo 0){
            cnt = if(set.contains(j * n + i)) 0 else cnt + 1
            dp[j][i] = min(dp[j][i], cnt)
            ans = max(dp[j][i], ans)
        }
    }
    return ans
}

//https://leetcode.cn/problems/maximum-units-on-a-truck/ 卡车上的最大单元数
fun maximumUnits(boxTypes: Array<IntArray>, truckSize: Int): Int {
    boxTypes.sortBy { -it[1] }
    var ans = 0
    var size = truckSize
    for ((num, units) in boxTypes){
        if(num >= size){
            ans += size * units
            break
        }
        ans += num * units
        size -= num
    }
    return ans
}

//https://leetcode.cn/problems/soup-servings/ 分汤
fun soupServings(n: Int): Double {
    val tot = (n + 24) / 25
    if(tot >= 179) return 1.0
    val dp = Array(tot + 1){DoubleArray(tot + 1)}.apply {
        for (i in 1..tot){
            this[0][i] = 1.0
        }
        this[0][0] = 0.5
    }
    //(4, 0) (3, 1) (2, 2) (1, 3)
    for (i in 1..tot){
        for (j in 1..tot){
            dp[i][j] = 0.25 * (dp[max(0, i - 4)][j] + dp[max(0, i - 3)][max(0, j - 1)]
                    + dp[max(i - 2, 0)][max(j - 2, 0)] + dp[max(i - 1, 0)][max(j - 3, 0)])
        }
    }
    return dp[tot][tot]
}

//https://leetcode.cn/problems/expressive-words/ 情感丰富的文字
fun expressiveWords(s: String, words: Array<String>): Int {
    var ans = 0
    for (word in words){
        if (expand(s, word))
            ans++
    }
    return ans
}

private fun expand(s: String, s1: String): Boolean{
    var l0 = 0
    var l1 = 0
    while (l0 < s.length && l1 < s1.length) {
        if (s[l0] != s1[l1]) return false
        var c = s[l0]
        var cnt0 = 0
        while (l0 < s.length && s[l0] == c){
            cnt0++
            l0++
        }
        var cnt1 = 0
        while (l1 < s1.length && s1[l1] == c){
            cnt1++
            l1++
        }
        if (cnt0 < cnt1) return false
        if (cnt0 != cnt1 && cnt0 < 3) return false
    }
    return l0 == s.length && l1 == s1.length
}
