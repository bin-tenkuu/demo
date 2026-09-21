package demo.util;

import java.util.HashMap;

public class RadixTrieTree<T> {
    private static class Node<T> {
        private T value;
        private final HashMap<String, Node<T>> children = new HashMap<>();

        public boolean isLeaf() {
            return value != null;
        }
    }

    private final Node<T> root = new Node<>();

    public void insert(String key, T value) {
        if (key == null) {
            throw new IllegalArgumentException("key cannot be null");
        }
    }

    public boolean search(String word) {
        return false;
    }

    public void delete(String word) {
    }

    private Node<T> delete(Node<T> current, String word) {
        return null;
    }

    public static void main(String[] args) {
        RadixTrieTree<Void> tree = new RadixTrieTree<>();
        tree.insert("test", null);
        tree.insert("water", null);
        tree.insert("slow", null);
        tree.insert("slower", null);
        tree.insert("team", null);
        tree.insert("tester", null);
        tree.insert("t", null);
        tree.insert("toast", null);

        System.out.println(tree.search("te"));

        tree.delete("test");

        System.out.println(tree.search("te"));
    }
}
