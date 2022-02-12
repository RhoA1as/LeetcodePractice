package com.oasis;

import java.util.ArrayList;
import java.util.List;

public class Test {
    public static void main(String[] args) {
        ArrayList<Student> list = new ArrayList<>();
        list.add(1,new Student("lv"));
        List<Student> temp = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            Student student = list.get(i);
            student.name = "z";
            temp.add(student);
            temp.add(1,student);
        }
        System.out.println(list.get(0).name);
    }
}
class Student{
    String name;
    public Student(String name){
        this.name = name;
    }
}