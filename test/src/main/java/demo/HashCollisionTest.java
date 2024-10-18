package demo;

import lombok.val;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * hash 碰撞测试
 * @author bin
 * @version 1.0.0
 * @since 2024/10/18
 */
public class HashCollisionTest {
    /**
     * 种子数据：两个长度为 2 的 hashCode 一样的字符串
     */
    private static final String[] SEED = {"Aa", "BB"};

    /**
     * 生成 2 的 n 次方个 HashCode 一样的字符串的集合
     */
    public static List<String> hashCodeSomeList(int n) {
        List<String> initList = new ArrayList<>(Arrays.asList(SEED));
        for (int i = 1; i < n; i++) {
            initList = createByList(initList);
        }
        return initList;
    }

    public static List<String> createByList(List<String> list) {
        val result = new ArrayList<String>();
        for (String s : SEED) {
            for (String str : list) {
                result.add(s + str);
            }
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println("Aa = " + "Aa".hashCode());
        System.out.println("BB = " + "BB".hashCode());

        System.out.println("Ab = " + "Ab".hashCode());
        System.out.println("BC = " + "BC".hashCode());

        System.out.println("Ac = " + "Ac".hashCode());
        System.out.println("BD = " + "BD".hashCode());

        System.out.println("AaAa = " + "AaAa".hashCode());
        System.out.println("BBBB = " + "BBBB".hashCode());
        System.out.println("AaBB = " + "AaBB".hashCode());
        System.out.println("BBAa = " + "BBAa".hashCode());

        System.out.println("AaAaAa = " + "AaAaAa".hashCode());
        System.out.println("AaAaBB = " + "AaAaBB".hashCode());
        System.out.println("AaBBAa = " + "AaBBAa".hashCode());
        System.out.println("AaBBBB = " + "AaBBBB".hashCode());
        System.out.println("BBAaAa = " + "BBAaAa".hashCode());
        System.out.println("BBAaBB = " + "BBAaBB".hashCode());
        System.out.println("BBBBAa = " + "BBBBAa".hashCode());
        System.out.println("BBBBBB = " + "BBBBBB".hashCode());
    }
}
