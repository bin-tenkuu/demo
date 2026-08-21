package demo.util;

import java.util.HashMap;

/**
 * 前缀树
 * <p>
 * 在树中存储字符串，每条边代表一个字符——共享前缀共享节点，实现按键长度 O(k) 查找。
 *
 * @author bin
 * @since 2026/08/21
 */
public class Trie {
    private static final class TrieNode {
        private final HashMap<Byte, TrieNode> children = new HashMap<>();
        private boolean isEnd = false;

        @Override
        public String toString() {
            return "TrieNode(" +
                    "children.size=" + children.size() + ", " +
                    "isEnd=" + isEnd + ')';
        }
    }

    private final TrieNode root = new TrieNode();

    public void insert(String word) {
        var node = root;
        for (byte b : word.getBytes()) {
            node = node.children.computeIfAbsent(b, _ -> new TrieNode());
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        var node = root;
        for (byte b : word.getBytes()) {
            node = node.children.get(b);
            if (node == null) {
                return false;
            }
        }
        return node.isEnd;
    }

    private TrieNode findNode(String word) {
        var node = root;
        for (byte b : word.getBytes()) {
            node = node.children.get(b);
            if (node == null) {
                return null;
            }
        }
        return node;
    }

    public boolean startsWith(String prefix) {
        return findNode(prefix) != null;
    }
}
