package demo.util;

import java.util.Map;
import java.util.TreeMap;

/**
 * 前缀树
 * <p>
 * 在树中存储字符串，每条边代表一个字符——共享前缀共享节点，实现按键长度 O(k) 查找。
 *
 * @author bin
 * @since 2026/08/21
 */
@SuppressWarnings("unused")
public class TrieTree<T> {
    private static final class Node<T> {
        private final TreeMap<Character, Node<T>> children = new TreeMap<>();
        private T value;

        public boolean isLeaf() {
            return value != null;
        }
    }

    private final Node<T> root = new Node<>();

    public void insert(String key, T value) {
        if (key == null) {
            throw new IllegalArgumentException("key cannot be null");
        }

        var node = root;
        for (int i = 0, size = key.length(); i < size; i++) {
            var ch = key.charAt(i);
            node = node.children.computeIfAbsent(ch, k -> new Node<>());
        }

        node.value = value;
    }

    public T search(String key) {
        var node = findNode(key);
        if (node == null) {
            return null;
        }
        return node.value;
    }

    public boolean startsWith(String prefix) {
        return findNode(prefix) != null;
    }

    private Node<T> findNode(String key) {
        if (key == null) {
            return null;
        }

        var node = root;
        for (int i = 0, size = key.length(); i < size; i++) {
            var ch = key.charAt(i);
            node = node.children.get(ch);
            if (node == null) {
                return null;
            }
        }
        return node;
    }

    @Override
    public String toString() {
        var sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }

    public void toString(StringBuilder sb) {
        sb.append("<root>");
        if (root.isLeaf()) {
            sb.append("●");
        }
        appendTree(root, sb, "", true);
    }

    private void appendTree(Node<T> node, StringBuilder sb, String prefix, boolean isLeaf) {
        var children = node.children;
        if (!isLeaf && children.size() == 1) {
            var entry = children.firstEntry();
            appendTree(entry, sb, prefix);
            return;
        }
        for (var entry : children.entrySet()) {
            sb.append('\n').append(prefix).append("\t");
            appendTree(entry, sb, prefix + "\t");
        }
    }

    private void appendTree(Map.Entry<Character, Node<T>> entry, StringBuilder sb, String prefix) {
        sb.append(entry.getKey());
        var child = entry.getValue();
        if (child.isLeaf()) {
            sb.append(" ●");
        }
        if (!child.children.isEmpty()) {
            appendTree(child, sb, prefix, child.isLeaf());
        }
    }

    public static void main(String[] args) {
        var tree = new TrieTree<Void>();
        tree.insert("card", null);
        tree.insert("care", null);
        tree.insert("cat", null);
        // tree.insert("cut", null);
        // tree.insert("dog", null);
        System.out.println(tree);
    }
}
