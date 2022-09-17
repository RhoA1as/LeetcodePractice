package com.oasis

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
