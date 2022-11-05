package com.oasis

import kotlin.math.abs
import kotlin.math.max

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
