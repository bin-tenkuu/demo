/*
 * Copyright (c) 1997, 2023, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package demo.test;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Hash table based implementation of the {@code Map} interface.  This
 * implementation provides all of the optional map operations, and permits
 * {@code null} values and the {@code null} key.  (The {@code HashMap}
 * class is roughly equivalent to {@code Hashtable}, except that it is
 * unsynchronized and permits nulls.)  This class makes no guarantees as to
 * the order of the map; in particular, it does not guarantee that the order
 * will remain constant over time.
 *
 * <p>This implementation provides constant-time performance for the basic
 * operations ({@code get} and {@code put}), assuming the hash function
 * disperses the elements properly among the buckets.  Iteration over
 * collection views requires time proportional to the "capacity" of the
 * {@code HashMap} instance (the number of buckets) plus its size (the number
 * of key-value mappings).  Thus, it's very important not to set the initial
 * capacity too high (or the load factor too low) if iteration performance is
 * important.
 *
 * <p>An instance of {@code HashMap} has two parameters that affect its
 * performance: <i>initial capacity</i> and <i>load factor</i>.  The
 * <i>capacity</i> is the number of buckets in the hash table, and the initial
 * capacity is simply the capacity at the time the hash table is created.  The
 * <i>load factor</i> is a measure of how full the hash table is allowed to
 * get before its capacity is automatically increased.  When the number of
 * entries in the hash table exceeds the product of the load factor and the
 * current capacity, the hash table is <i>rehashed</i> (that is, internal data
 * structures are rebuilt) so that the hash table has approximately twice the
 * number of buckets.
 *
 * <p>As a general rule, the default load factor (.75) offers a good
 * tradeoff between time and space costs.  Higher values decrease the
 * space overhead but increase the lookup cost (reflected in most of
 * the operations of the {@code HashMap} class, including
 * {@code get} and {@code put}).  The expected number of entries in
 * the map and its load factor should be taken into account when
 * setting its initial capacity, so as to minimize the number of
 * rehash operations.  If the initial capacity is greater than the
 * maximum number of entries divided by the load factor, no rehash
 * operations will ever occur.
 *
 * <p>If many mappings are to be stored in a {@code HashMap}
 * instance, creating it with a sufficiently large capacity will allow
 * the mappings to be stored more efficiently than letting it perform
 * automatic rehashing as needed to grow the table.  Note that using
 * many keys with the same {@code hashCode()} is a sure way to slow
 * down performance of any hash table. To ameliorate impact, when keys
 * are {@link Comparable}, this class may use comparison order among
 * keys to help break ties.
 *
 * <p><strong>Note that this implementation is not synchronized.</strong>
 * If multiple threads access a hash map concurrently, and at least one of
 * the threads modifies the map structurally, it <i>must</i> be
 * synchronized externally.  (A structural modification is any operation
 * that adds or deletes one or more mappings; merely changing the value
 * associated with a key that an instance already contains is not a
 * structural modification.)  This is typically accomplished by
 * synchronizing on some object that naturally encapsulates the map.
 * <p>
 * If no such object exists, the map should be "wrapped" using the
 * {@link Collections#synchronizedMap Collections.synchronizedMap}
 * method.  This is best done at creation time, to prevent accidental
 * unsynchronized access to the map:<pre>
 *   Map m = Collections.synchronizedMap(new HashMap(...));</pre>
 *
 * <p>The iterators returned by all of this class's "collection view methods"
 * are <i>fail-fast</i>: if the map is structurally modified at any time after
 * the iterator is created, in any way except through the iterator's own
 * {@code remove} method, the iterator will throw a
 * {@link ConcurrentModificationException}.  Thus, in the face of concurrent
 * modification, the iterator fails quickly and cleanly, rather than risking
 * arbitrary, non-deterministic behavior at an undetermined time in the
 * future.
 *
 * <p>Note that the fail-fast behavior of an iterator cannot be guaranteed
 * as it is, generally speaking, impossible to make any hard guarantees in the
 * presence of unsynchronized concurrent modification.  Fail-fast iterators
 * throw {@code ConcurrentModificationException} on a best-effort basis.
 * Therefore, it would be wrong to write a program that depended on this
 * exception for its correctness: <i>the fail-fast behavior of iterators
 * should be used only to detect bugs.</i>
 *
 * <p>This class is a member of the
 * <a href="{@docRoot}/java.base/java/util/package-summary.html#CollectionsFramework">
 * Java Collections Framework</a>.
 *
 * @param <V> the type of mapped values
 * @author Doug Lea
 * @author Josh Bloch
 * @author Arthur van Hoff
 * @author Neal Gafter
 * @see Object#hashCode()
 * @see Collection
 * @see Map
 * @see TreeMap
 * @see Hashtable
 * @since 1.2
 */
@SuppressWarnings({"DuplicatedCode"})
public final class LongArrayHashMap<V> {

    /**
     * The default initial capacity - MUST be a power of two.
     */
    private static final int DEFAULT_INITIAL_CAPACITY = 1 << 4; // aka 16

    /**
     * The maximum capacity, used if a higher value is implicitly specified
     * by either of the constructors with arguments.
     * MUST be a power of two <= 1<<30.
     */
    private static final int MAXIMUM_CAPACITY = 1 << 30;

    /**
     * The load factor used when none specified in constructor.
     */
    private static final float DEFAULT_LOAD_FACTOR = 0.75f;

    /**
     * The bin count threshold for using a tree rather than list for a
     * bin.  Bins are converted to trees when adding an element to a
     * bin with at least this many nodes. The value must be greater
     * than 2 and should be at least 8 to mesh with assumptions in
     * tree removal about conversion back to plain bins upon
     * shrinkage.
     */
    private static final int TREEIFY_THRESHOLD = 8;

    /**
     * The bin count threshold for untreeifying a (split) bin during a
     * resize operation. Should be less than TREEIFY_THRESHOLD, and at
     * most 6 to mesh with shrinkage detection under removal.
     */
    private static final int UNTREEIFY_THRESHOLD = 6;

    /**
     * The smallest table capacity for which bins may be treeified.
     * (Otherwise the table is resized if too many nodes in a bin.)
     * Should be at least 4 * TREEIFY_THRESHOLD to avoid conflicts
     * between resizing and treeification thresholds.
     */
    private static final int MIN_TREEIFY_CAPACITY = 64;

    /**
     * Basic hash bin node, used for most entries.  (See below for
     * TreeNode subclass, and in LinkedHashMap for its Entry subclass.)
     */
    public static class Node<V> {
        final int hash;
        final long[] key;
        V value;
        Node<V> next;

        Node(int hash, long[] key, V value, Node<V> next) {
            this.hash = hash;
            this.key = key;
            this.value = value;
            this.next = next;
        }

        public final long[] getKey() {
            return key;
        }

        public final V getValue() {
            return value;
        }

        public final String toString() {
            return LongArrayHashMap.toString(key) + "=" + value;
        }

        public final int hashCode() {
            return hash(key) ^ Objects.hashCode(value);
        }

        public final V setValue(V newValue) {
            V oldValue = value;
            value = newValue;
            return oldValue;
        }

        public final boolean equals(Object o) {
            if (o == this) {
                return true;
            }

            return o instanceof Node<?> e
                   && Arrays.equals(key, e.getKey())
                   && Objects.equals(value, e.getValue());
        }
    }

    /* ---------------- Static utilities -------------- */

    /**
     * Computes key.hashCode() and spreads (XORs) higher bits of hash
     * to lower.  Because the table uses power-of-two masking, sets of
     * hashes that vary only in bits above the current mask will
     * always collide. (Among known examples are sets of Float keys
     * holding consecutive whole numbers in small tables.)  So we
     * apply a transform that spreads the impact of higher bits
     * downward. There is a tradeoff between speed, utility, and
     * quality of bit-spreading. Because many common sets of hashes
     * are already reasonably distributed (so don't benefit from
     * spreading), and because we use trees to handle large sets of
     * collisions in bins, we just XOR some shifted bits in the
     * cheapest possible way to reduce systematic lossage, as well as
     * to incorporate impact of the highest bits that would otherwise
     * never be used in index calculations because of table bounds.
     */
    private static int hash(long[] key) {
        if (key == null) {
            return 0;
        }
        int result = 0;
        for (long element : key) {
            result = result ^ Long.hashCode(element);
        }
        return result;
    }

    private static boolean equal(long[] k, long[] x) {
        return Arrays.equals(k, x);
    }

    private static String toString(long[] k) {
        return Arrays.toString(k);
    }

    /**
     * Returns k.compareTo(x) if x matches kc (k's screened comparable
     * class), else 0.
     */
    private static int compareComparables(long[] k, long[] x) {
        return Arrays.compare(k, x);
    }

    /**
     * Returns a power of two size for the given target capacity.
     */
    private static int tableSizeFor(int cap) {
        int n = -1 >>> Integer.numberOfLeadingZeros(cap - 1);
        return n < 0 ? 1 : n >= MAXIMUM_CAPACITY ? MAXIMUM_CAPACITY : n + 1;
    }

    /* ---------------- Fields -------------- */

    /**
     * The table, initialized on first use, and resized as
     * necessary. When allocated, length is always a power of two.
     * (We also tolerate length zero in some operations to allow
     * bootstrapping mechanics that are currently not needed.)
     */
    private transient Node<V>[] table;

    /**
     * Holds cached entrySet(). Note that AbstractMap fields are used
     * for keySet() and values().
     */
    private transient Set<Node<V>> entrySet;
    private transient Set<long[]> keySet;
    private transient Collection<V> values;
    /**
     * The number of key-value mappings contained in this map.
     */
    private transient int size;

    /**
     * The number of times this HashMap has been structurally modified
     * Structural modifications are those that change the number of mappings in
     * the HashMap or otherwise modify its internal structure (e.g.,
     * rehash).  This field is used to make iterators on Collection-views of
     * the HashMap fail-fast.  (See ConcurrentModificationException).
     */
    private transient int modCount;

    /**
     * The next size value at which to resize (capacity * load factor).
     *
     * @serial
     */
    // (The javadoc description is true upon serialization.
    // Additionally, if the table array has not been allocated, this
    // field holds the initial array capacity, or zero signifying
    // DEFAULT_INITIAL_CAPACITY.)
    private int threshold;

    /**
     * The load factor for the hash table.
     *
     * @serial
     */
    private final float loadFactor;

    /* ---------------- Public operations -------------- */

    /**
     * Constructs an empty {@code HashMap} with the specified initial
     * capacity and load factor.
     *
     * @param initialCapacity the initial capacity
     * @param loadFactor the load factor
     * @throws IllegalArgumentException if the initial capacity is negative
     * or the load factor is nonpositive
     * @apiNote To create a {@code HashMap} with an initial capacity that accommodates
     * an expected number of mappings, use {@link #newHashMap(int) newHashMap}.
     */
    public LongArrayHashMap(int initialCapacity, float loadFactor) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Illegal initial capacity: " + initialCapacity);
        }
        if (initialCapacity > MAXIMUM_CAPACITY) {
            initialCapacity = MAXIMUM_CAPACITY;
        }
        if (loadFactor <= 0 || Float.isNaN(loadFactor)) {
            throw new IllegalArgumentException("Illegal load factor: " + loadFactor);
        }
        this.loadFactor = loadFactor;
        this.threshold = tableSizeFor(initialCapacity);
    }

    /**
     * Constructs an empty {@code HashMap} with the specified initial
     * capacity and the default load factor (0.75).
     *
     * @param initialCapacity the initial capacity.
     * @throws IllegalArgumentException if the initial capacity is negative.
     * @apiNote To create a {@code HashMap} with an initial capacity that accommodates
     * an expected number of mappings, use {@link #newHashMap(int) newHashMap}.
     */
    public LongArrayHashMap(int initialCapacity) {
        this(initialCapacity, DEFAULT_LOAD_FACTOR);
    }

    /**
     * Constructs an empty {@code HashMap} with the default initial capacity
     * (16) and the default load factor (0.75).
     */
    public LongArrayHashMap() {
        this.loadFactor = DEFAULT_LOAD_FACTOR; // all other fields defaulted
    }

    /**
     * Constructs a new {@code HashMap} with the same mappings as the
     * specified {@code Map}.  The {@code HashMap} is created with
     * default load factor (0.75) and an initial capacity sufficient to
     * hold the mappings in the specified {@code Map}.
     *
     * @param m the map whose mappings are to be placed in this map
     * @throws NullPointerException if the specified map is null
     */
    public LongArrayHashMap(Map<? extends long[], ? extends V> m) {
        this.loadFactor = DEFAULT_LOAD_FACTOR;
        putMapEntries(m);
    }

    /**
     * Implements Map.putAll and Map constructor.
     *
     * @param m the map
     */
    private void putMapEntries(Map<? extends long[], ? extends V> m) {
        int s = m.size();
        if (s > 0) {
            if (table == null) { // pre-size
                double dt = Math.ceil(s / (double) loadFactor);
                int t = dt < (double) MAXIMUM_CAPACITY ? (int) dt : MAXIMUM_CAPACITY;
                if (t > threshold) {
                    threshold = tableSizeFor(t);
                }
            } else {
                // Because of linked-list bucket constraints, we cannot
                // expand all at once, but can reduce total resize
                // effort by repeated doubling now vs later
                while (s > threshold && table.length < MAXIMUM_CAPACITY) {
                    resize();
                }
            }

            for (Map.Entry<? extends long[], ? extends V> e : m.entrySet()) {
                long[] key = e.getKey();
                V value = e.getValue();
                putVal(hash(key), key, value, false);
            }
        }
    }

    /**
     * Returns the number of key-value mappings in this map.
     *
     * @return the number of key-value mappings in this map
     */
    public int size() {
        return size;
    }

    /**
     * Returns {@code true} if this map contains no key-value mappings.
     *
     * @return {@code true} if this map contains no key-value mappings
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Returns the value to which the specified key is mapped,
     * or {@code null} if this map contains no mapping for the key.
     *
     * <p>More formally, if this map contains a mapping from a key
     * {@code k} to a value {@code v} such that {@code (key==null ? k==null :
     * key.equals(k))}, then this method returns {@code v}; otherwise
     * it returns {@code null}.  (There can be at most one such mapping.)
     *
     * <p>A return value of {@code null} does not <i>necessarily</i>
     * indicate that the map contains no mapping for the key; it's also
     * possible that the map explicitly maps the key to {@code null}.
     * The {@link #containsKey containsKey} operation may be used to
     * distinguish these two cases.
     *
     * @see #put(long[], Object)
     */
    public V get(long[] key) {
        Node<V> e;
        return (e = getNode(key)) == null ? null : e.value;
    }

    /**
     * Implements Map.get and related methods.
     *
     * @param key the key
     * @return the node, or null if none
     */
    private Node<V> getNode(long[] key) {
        Node<V>[] tab;
        Node<V> first, e;
        int n, hash;
        if ((tab = table) != null && (n = tab.length) > 0 &&
            (first = tab[n - 1 & (hash = hash(key))]) != null) {
            if (first.hash == hash && equal(key, first.key)) {
                return first;
            }
            e = first.next;
            if (e != null) {
                if (first instanceof TreeNode) {
                    return ((TreeNode<V>) first).getTreeNode(hash, key);
                }
                do {
                    if (e.hash == hash && equal(key, e.key)) {
                        return e;
                    }
                    e = e.next;
                } while (e != null);
            }
        }
        return null;
    }

    /**
     * Returns {@code true} if this map contains a mapping for the
     * specified key.
     *
     * @param key The key whose presence in this map is to be tested
     * @return {@code true} if this map contains a mapping for the specified
     * key.
     */
    public boolean containsKey(long[] key) {
        return getNode(key) != null;
    }

    /**
     * Associates the specified value with the specified key in this map.
     * If the map previously contained a mapping for the key, the old
     * value is replaced.
     *
     * @param key key with which the specified value is to be associated
     * @param value value to be associated with the specified key
     * @return the previous value associated with {@code key}, or
     * {@code null} if there was no mapping for {@code key}.
     * (A {@code null} return can also indicate that the map
     * previously associated {@code null} with {@code key}.)
     */
    public V put(long[] key, V value) {
        return putVal(hash(key), key, value, false);
    }

    /**
     * Implements Map.put and related methods.
     *
     * @param hash hash for key
     * @param key the key
     * @param value the value to put
     * @param onlyIfAbsent if true, don't change existing value
     * @return previous value, or null if none
     */
    private V putVal(int hash, long[] key, V value, boolean onlyIfAbsent) {
        Node<V>[] tab;
        Node<V> p;
        int n, i;
        if ((tab = table) == null || (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        p = tab[i = n - 1 & hash];
        if (p == null) {
            tab[i] = newNode(hash, key, value, null);
        } else {
            Node<V> e;
            if (p.hash == hash && equal(key, p.key)) {
                e = p;
            } else if (p instanceof TreeNode) {
                e = ((TreeNode<V>) p).putTreeVal(tab, hash, key, value);
            } else {
                for (int binCount = 0; ; ++binCount) {
                    e = p.next;
                    if (e == null) {
                        p.next = newNode(hash, key, value, null);
                        if (binCount >= TREEIFY_THRESHOLD - 1) { // -1 for 1st
                            treeifyBin(tab, hash);
                        }
                        break;
                    }
                    if (e.hash == hash &&
                        equal(key, e.key)) {
                        break;
                    }
                    p = e;
                }
            }
            if (e != null) { // existing mapping for key
                V oldValue = e.value;
                if (!onlyIfAbsent || oldValue == null) {
                    e.value = value;
                }
                return oldValue;
            }
        }
        ++modCount;
        if (++size > threshold) {
            resize();
        }
        return null;
    }

    /**
     * Initializes or doubles table size.  If null, allocates in
     * accord with initial capacity target held in field threshold.
     * Otherwise, because we are using power-of-two expansion, the
     * elements from each bin must either stay at same index, or move
     * with a power of two offset in the new table.
     *
     * @return the table
     */
    private Node<V>[] resize() {
        Node<V>[] oldTab = table;
        int oldCap = oldTab == null ? 0 : oldTab.length;
        int oldThr = threshold;
        int newCap, newThr = 0;
        if (oldCap > 0) {
            if (oldCap >= MAXIMUM_CAPACITY) {
                threshold = Integer.MAX_VALUE;
                return oldTab;
            } else if ((newCap = oldCap << 1) < MAXIMUM_CAPACITY &&
                       oldCap >= DEFAULT_INITIAL_CAPACITY) {
                newThr = oldThr << 1; // double threshold
            }
        } else if (oldThr > 0) // initial capacity was placed in threshold
        {
            newCap = oldThr;
        } else {               // zero initial threshold signifies using defaults
            newCap = DEFAULT_INITIAL_CAPACITY;
            newThr = (int) (DEFAULT_LOAD_FACTOR * DEFAULT_INITIAL_CAPACITY);
        }
        if (newThr == 0) {
            float ft = (float) newCap * loadFactor;
            newThr = newCap < MAXIMUM_CAPACITY && ft < (float) MAXIMUM_CAPACITY ?
                    (int) ft : Integer.MAX_VALUE;
        }
        threshold = newThr;
        @SuppressWarnings({"unchecked"})
        Node<V>[] newTab = (Node<V>[]) new Node[newCap];
        table = newTab;
        if (oldTab != null) {
            for (int j = 0; j < oldCap; ++j) {
                Node<V> e;
                if ((e = oldTab[j]) != null) {
                    oldTab[j] = null;
                    if (e.next == null) {
                        newTab[e.hash & newCap - 1] = e;
                    } else if (e instanceof TreeNode) {
                        ((TreeNode<V>) e).split(newTab, j, oldCap);
                    } else { // preserve order
                        Node<V> loHead = null, loTail = null;
                        Node<V> hiHead = null, hiTail = null;
                        Node<V> next;
                        do {
                            next = e.next;
                            if ((e.hash & oldCap) == 0) {
                                if (loTail == null) {
                                    loHead = e;
                                } else {
                                    loTail.next = e;
                                }
                                loTail = e;
                            } else {
                                if (hiTail == null) {
                                    hiHead = e;
                                } else {
                                    hiTail.next = e;
                                }
                                hiTail = e;
                            }
                        } while ((e = next) != null);
                        if (loTail != null) {
                            loTail.next = null;
                            newTab[j] = loHead;
                        }
                        if (hiTail != null) {
                            hiTail.next = null;
                            newTab[j + oldCap] = hiHead;
                        }
                    }
                }
            }
        }
        return newTab;
    }

    /**
     * Replaces all linked nodes in bin at index for given hash unless
     * table is too small, in which case resizes instead.
     */
    private void treeifyBin(Node<V>[] tab, int hash) {
        int n, index;
        Node<V> e;
        if (tab == null || (n = tab.length) < MIN_TREEIFY_CAPACITY) {
            resize();
        } else if ((e = tab[index = n - 1 & hash]) != null) {
            TreeNode<V> hd = null, tl = null;
            do {
                TreeNode<V> p = replacementTreeNode(e);
                if (tl == null) {
                    hd = p;
                } else {
                    p.prev = tl;
                    tl.next = p;
                }
                tl = p;
            } while ((e = e.next) != null);
            tab[index] = hd;
            hd.treeify(tab);
        }
    }

    /**
     * Copies all of the mappings from the specified map to this map.
     * These mappings will replace any mappings that this map had for
     * any of the keys currently in the specified map.
     *
     * @param m mappings to be stored in this map
     * @throws NullPointerException if the specified map is null
     */
    public void putAll(Map<? extends long[], ? extends V> m) {
        putMapEntries(m);
    }

    /**
     * Removes the mapping for the specified key from this map if present.
     *
     * @param key key whose mapping is to be removed from the map
     * @return the previous value associated with {@code key}, or
     * {@code null} if there was no mapping for {@code key}.
     * (A {@code null} return can also indicate that the map
     * previously associated {@code null} with {@code key}.)
     */
    public V remove(long[] key) {
        Node<V> e;
        return (e = removeNode(hash(key), key, null, false, true)) == null ?
                null : e.value;
    }

    /**
     * Implements Map.remove and related methods.
     *
     * @param hash hash for key
     * @param key the key
     * @param value the value to match if matchValue, else ignored
     * @param matchValue if true only remove if value is equal
     * @param movable if false do not move other nodes while removing
     * @return the node, or null if none
     */
    private Node<V> removeNode(int hash, long[] key, Object value,
            boolean matchValue, boolean movable) {
        Node<V>[] tab;
        Node<V> p;
        int n, index;
        if ((tab = table) != null && (n = tab.length) > 0 &&
            (p = tab[index = n - 1 & hash]) != null) {
            Node<V> node = null, e;
            V v;
            if (p.hash == hash && equal(key, p.key)) {
                node = p;
            } else if ((e = p.next) != null) {
                if (p instanceof TreeNode) {
                    node = ((TreeNode<V>) p).getTreeNode(hash, key);
                } else {
                    do {
                        if (e.hash == hash && equal(key, e.key)) {
                            node = e;
                            break;
                        }
                        p = e;
                    } while ((e = e.next) != null);
                }
            }
            if (node != null && (!matchValue || (v = node.value) == value ||
                                 value != null && value.equals(v))) {
                if (node instanceof TreeNode) {
                    ((TreeNode<V>) node).removeTreeNode(tab, movable);
                } else if (node == p) {
                    tab[index] = node.next;
                } else {
                    p.next = node.next;
                }
                ++modCount;
                --size;
                return node;
            }
        }
        return null;
    }

    /**
     * Removes all of the mappings from this map.
     * The map will be empty after this call returns.
     */
    public void clear() {
        Node<V>[] tab;
        modCount++;
        if ((tab = table) != null && size > 0) {
            size = 0;
            Arrays.fill(tab, null);
        }
    }

    /**
     * Returns {@code true} if this map maps one or more keys to the
     * specified value.
     *
     * @param value value whose presence in this map is to be tested
     * @return {@code true} if this map maps one or more keys to the
     * specified value
     */
    public boolean containsValue(Object value) {
        Node<V>[] tab;
        V v;
        if ((tab = table) != null && size > 0) {
            for (Node<V> e : tab) {
                for (; e != null; e = e.next) {
                    if ((v = e.value) == value ||
                        value != null && value.equals(v)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Returns a {@link Set} view of the keys contained in this map.
     * The set is backed by the map, so changes to the map are
     * reflected in the set, and vice-versa.  If the map is modified
     * while an iteration over the set is in progress (except through
     * the iterator's own {@code remove} operation), the results of
     * the iteration are undefined.  The set supports element removal,
     * which removes the corresponding mapping from the map, via the
     * {@code Iterator.remove}, {@code Set.remove},
     * {@code removeAll}, {@code retainAll}, and {@code clear}
     * operations.  It does not support the {@code add} or {@code addAll}
     * operations.
     *
     * @return a set view of the keys contained in this map
     */
    public Set<long[]> keySet() {
        Set<long[]> ks = keySet;
        if (ks == null) {
            ks = new KeySet();
            keySet = ks;
        }
        return ks;
    }

    /**
     * Prepares the array for {@link Collection#toArray(Object[])} implementation.
     * If supplied array is smaller than this map size, a new array is allocated.
     * If supplied array is bigger than this map size, a null is written at size index.
     *
     * @param a an original array passed to {@code toArray()} method
     * @param <T> type of array elements
     * @return an array ready to be filled and returned from {@code toArray()} method.
     */
    @SuppressWarnings("unchecked")
    private <T> T[] prepareArray(T[] a) {
        int size = this.size;
        if (a.length < size) {
            return (T[]) java.lang.reflect.Array
                    .newInstance(a.getClass().getComponentType(), size);
        }
        if (a.length > size) {
            a[size] = null;
        }
        return a;
    }

    /**
     * Fills an array with this map keys and returns it. This method assumes
     * that input array is big enough to fit all the keys.
     *
     * @return supplied array
     */
    private long[][] keysToArray() {
        long[][] r = new long[size][];
        Node<V>[] tab;
        int idx = 0;
        if (size > 0 && (tab = table) != null) {
            for (Node<V> e : tab) {
                for (; e != null; e = e.next) {
                    r[idx++] = e.key;
                }
            }
        }
        return r;
    }

    /**
     * Fills an array with this map values and returns it. This method assumes
     * that input array is big enough to fit all the values. Use
     * {@link #prepareArray(Object[])} to ensure this.
     *
     * @param a an array to fill
     * @param <T> type of array elements
     * @return supplied array
     */
    private <T> T[] valuesToArray(T[] a) {
        Object[] r = a;
        Node<V>[] tab;
        int idx = 0;
        if (size > 0 && (tab = table) != null) {
            for (Node<V> e : tab) {
                for (; e != null; e = e.next) {
                    r[idx++] = e.value;
                }
            }
        }
        return a;
    }

    private final class KeySet extends AbstractSet<long[]> {
        public int size() {
            return size;
        }

        public void clear() {
            LongArrayHashMap.this.clear();
        }

        public Iterator<long[]> iterator() {
            return new KeyIterator();
        }

        public boolean contains(long[] o) {
            return containsKey(o);
        }

        public boolean remove(long[] key) {
            return removeNode(hash(key), key, null, false, true) != null;
        }

        public Spliterator<long[]> spliterator() {
            return new KeySpliterator<>(LongArrayHashMap.this, 0, -1, 0, 0);
        }

        public Object[] toArray() {
            return keysToArray();
        }

        public <T> T[] toArray(T[] a) {
            if (a instanceof long[][]) {
                return (T[]) keysToArray();
            } else {
                return a;
            }
        }

        public void forEach(Consumer<? super long[]> action) {
            Node<V>[] tab;
            if (action == null) {
                throw new NullPointerException();
            }
            if (size > 0 && (tab = table) != null) {
                int mc = modCount;
                for (Node<V> e : tab) {
                    for (; e != null; e = e.next) {
                        action.accept(e.key);
                    }
                }
                if (modCount != mc) {
                    throw new ConcurrentModificationException();
                }
            }
        }
    }

    /**
     * Returns a {@link Collection} view of the values contained in this map.
     * The collection is backed by the map, so changes to the map are
     * reflected in the collection, and vice-versa.  If the map is
     * modified while an iteration over the collection is in progress
     * (except through the iterator's own {@code remove} operation),
     * the results of the iteration are undefined.  The collection
     * supports element removal, which removes the corresponding
     * mapping from the map, via the {@code Iterator.remove},
     * {@code Collection.remove}, {@code removeAll},
     * {@code retainAll} and {@code clear} operations.  It does not
     * support the {@code add} or {@code addAll} operations.
     *
     * @return a view of the values contained in this map
     */
    public Collection<V> values() {
        Collection<V> vs = values;
        if (vs == null) {
            vs = new Values();
            values = vs;
        }
        return vs;
    }

    private final class Values extends AbstractCollection<V> {
        public int size() {
            return size;
        }

        public void clear() {
            LongArrayHashMap.this.clear();
        }

        public Iterator<V> iterator() {
            return new ValueIterator();
        }

        public boolean contains(Object o) {
            return containsValue(o);
        }

        public Spliterator<V> spliterator() {
            return new ValueSpliterator<>(LongArrayHashMap.this, 0, -1, 0, 0);
        }

        public Object[] toArray() {
            return valuesToArray(new Object[size]);
        }

        public <T> T[] toArray(T[] a) {
            return valuesToArray(prepareArray(a));
        }

        public void forEach(Consumer<? super V> action) {
            Node<V>[] tab;
            if (action == null) {
                throw new NullPointerException();
            }
            if (size > 0 && (tab = table) != null) {
                int mc = modCount;
                for (Node<V> e : tab) {
                    for (; e != null; e = e.next) {
                        action.accept(e.value);
                    }
                }
                if (modCount != mc) {
                    throw new ConcurrentModificationException();
                }
            }
        }
    }

    /**
     * Returns a {@link Set} view of the mappings contained in this map.
     * The set is backed by the map, so changes to the map are
     * reflected in the set, and vice-versa.  If the map is modified
     * while an iteration over the set is in progress (except through
     * the iterator's own {@code remove} operation, or through the
     * {@code setValue} operation on a map entry returned by the
     * iterator) the results of the iteration are undefined.  The set
     * supports element removal, which removes the corresponding
     * mapping from the map, via the {@code Iterator.remove},
     * {@code Set.remove}, {@code removeAll}, {@code retainAll} and
     * {@code clear} operations.  It does not support the
     * {@code add} or {@code addAll} operations.
     *
     * @return a set view of the mappings contained in this map
     */
    public Set<Node<V>> entrySet() {
        Set<Node<V>> es;
        return (es = entrySet) == null ? (entrySet = new EntrySet()) : es;
    }

    private final class EntrySet extends AbstractSet<Node<V>> {
        public int size() {
            return size;
        }

        public void clear() {
            LongArrayHashMap.this.clear();
        }

        public Iterator<Node<V>> iterator() {
            return new EntryIterator();
        }

        public boolean contains(Object o) {
            if (!(o instanceof Node<?> e)) {
                return false;
            }
            long[] key = e.getKey();
            Node<V> candidate = getNode(key);
            return candidate != null && candidate.equals(e);
        }

        public boolean remove(Object o) {
            if (o instanceof Node<?> e) {
                long[] key = e.getKey();
                Object value = e.getValue();
                return removeNode(hash(key), key, value, true, true) != null;
            }
            return false;
        }

        public Spliterator<Node<V>> spliterator() {
            return new EntrySpliterator<>(LongArrayHashMap.this, 0, -1, 0, 0);
        }

        public void forEach(Consumer<? super Node<V>> action) {
            Node<V>[] tab;
            if (action == null) {
                throw new NullPointerException();
            }
            if (size > 0 && (tab = table) != null) {
                int mc = modCount;
                for (Node<V> e : tab) {
                    for (; e != null; e = e.next) {
                        action.accept(e);
                    }
                }
                if (modCount != mc) {
                    throw new ConcurrentModificationException();
                }
            }
        }
    }

    // Overrides of JDK8 Map extension methods

    public V getOrDefault(long[] key, V defaultValue) {
        Node<V> e;
        return (e = getNode(key)) == null ? defaultValue : e.value;
    }

    public V putIfAbsent(long[] key, V value) {
        return putVal(hash(key), key, value, true);
    }

    public boolean remove(long[] key, Object value) {
        return removeNode(hash(key), key, value, true, true) != null;
    }

    public boolean replace(long[] key, V oldValue, V newValue) {
        Node<V> e;
        V v;
        if ((e = getNode(key)) != null &&
            ((v = e.value) == oldValue || v != null && v.equals(oldValue))) {
            e.value = newValue;
            return true;
        }
        return false;
    }

    public V replace(long[] key, V value) {
        Node<V> e;
        if ((e = getNode(key)) != null) {
            V oldValue = e.value;
            e.value = value;
            return oldValue;
        }
        return null;
    }

    /**
     * {@inheritDoc}
     *
     * <p>This method will, on a best-effort basis, throw a
     * {@link ConcurrentModificationException} if it is detected that the
     * mapping function modifies this map during computation.
     *
     * @throws ConcurrentModificationException if it is detected that the
     * mapping function modified this map
     */
    public V computeIfAbsent(long[] key,
            Function<? super long[], ? extends V> mappingFunction) {
        if (mappingFunction == null) {
            throw new NullPointerException();
        }
        int hash = hash(key);
        Node<V>[] tab;
        Node<V> first;
        int n, i;
        int binCount = 0;
        TreeNode<V> t = null;
        Node<V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        if ((first = tab[i = n - 1 & hash]) != null) {
            if (first instanceof TreeNode) {
                old = (t = (TreeNode<V>) first).getTreeNode(hash, key);
            } else {
                Node<V> e = first;
                do {
                    if (e.hash == hash &&
                        equal(key, e.key)) {
                        old = e;
                        break;
                    }
                    ++binCount;
                } while ((e = e.next) != null);
            }
            V oldValue;
            if (old != null && (oldValue = old.value) != null) {
                return oldValue;
            }
        }
        int mc = modCount;
        V v = mappingFunction.apply(key);
        if (mc != modCount) {
            throw new ConcurrentModificationException();
        }
        if (v == null) {
            return null;
        } else if (old != null) {
            old.value = v;
            return v;
        } else if (t != null) {
            t.putTreeVal(tab, hash, key, v);
        } else {
            tab[i] = newNode(hash, key, v, first);
            if (binCount >= TREEIFY_THRESHOLD - 1) {
                treeifyBin(tab, hash);
            }
        }
        modCount = mc + 1;
        ++size;
        return v;
    }

    /**
     * {@inheritDoc}
     *
     * <p>This method will, on a best-effort basis, throw a
     * {@link ConcurrentModificationException} if it is detected that the
     * remapping function modifies this map during computation.
     *
     * @throws ConcurrentModificationException if it is detected that the
     * remapping function modified this map
     */
    public V computeIfPresent(long[] key,
            BiFunction<? super long[], ? super V, ? extends V> remappingFunction) {
        if (remappingFunction == null) {
            throw new NullPointerException();
        }
        Node<V> e;
        V oldValue;
        if ((e = getNode(key)) != null &&
            (oldValue = e.value) != null) {
            int mc = modCount;
            V v = remappingFunction.apply(key, oldValue);
            if (mc != modCount) {
                throw new ConcurrentModificationException();
            }
            if (v != null) {
                e.value = v;
                return v;
            } else {
                int hash = hash(key);
                removeNode(hash, key, null, false, true);
            }
        }
        return null;
    }

    /**
     * {@inheritDoc}
     *
     * <p>This method will, on a best-effort basis, throw a
     * {@link ConcurrentModificationException} if it is detected that the
     * remapping function modifies this map during computation.
     *
     * @throws ConcurrentModificationException if it is detected that the
     * remapping function modified this map
     */
    public V compute(long[] key,
            BiFunction<? super long[], ? super V, ? extends V> remappingFunction) {
        if (remappingFunction == null) {
            throw new NullPointerException();
        }
        int hash = hash(key);
        Node<V>[] tab;
        Node<V> first;
        int n, i;
        int binCount = 0;
        TreeNode<V> t = null;
        Node<V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        if ((first = tab[i = n - 1 & hash]) != null) {
            if (first instanceof TreeNode) {
                old = (t = (TreeNode<V>) first).getTreeNode(hash, key);
            } else {
                Node<V> e = first;
                do {
                    if (e.hash == hash &&
                        equal(key, e.key)) {
                        old = e;
                        break;
                    }
                    ++binCount;
                } while ((e = e.next) != null);
            }
        }
        V oldValue = old == null ? null : old.value;
        int mc = modCount;
        V v = remappingFunction.apply(key, oldValue);
        if (mc != modCount) {
            throw new ConcurrentModificationException();
        }
        if (old != null) {
            if (v != null) {
                old.value = v;
            } else {
                removeNode(hash, key, null, false, true);
            }
        } else if (v != null) {
            if (t != null) {
                t.putTreeVal(tab, hash, key, v);
            } else {
                tab[i] = newNode(hash, key, v, first);
                if (binCount >= TREEIFY_THRESHOLD - 1) {
                    treeifyBin(tab, hash);
                }
            }
            modCount = mc + 1;
            ++size;
        }
        return v;
    }

    /**
     * {@inheritDoc}
     *
     * <p>This method will, on a best-effort basis, throw a
     * {@link ConcurrentModificationException} if it is detected that the
     * remapping function modifies this map during computation.
     *
     * @throws ConcurrentModificationException if it is detected that the
     * remapping function modified this map
     */
    public V merge(long[] key, V value,
            BiFunction<? super V, ? super V, ? extends V> remappingFunction) {
        if (value == null || remappingFunction == null) {
            throw new NullPointerException();
        }
        int hash = hash(key);
        Node<V>[] tab;
        Node<V> first;
        int n, i;
        int binCount = 0;
        TreeNode<V> t = null;
        Node<V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        if ((first = tab[i = n - 1 & hash]) != null) {
            if (first instanceof TreeNode) {
                old = (t = (TreeNode<V>) first).getTreeNode(hash, key);
            } else {
                Node<V> e = first;
                do {
                    if (e.hash == hash &&
                        equal(key, e.key)) {
                        old = e;
                        break;
                    }
                    ++binCount;
                } while ((e = e.next) != null);
            }
        }
        if (old != null) {
            V v;
            if (old.value != null) {
                int mc = modCount;
                v = remappingFunction.apply(old.value, value);
                if (mc != modCount) {
                    throw new ConcurrentModificationException();
                }
            } else {
                v = value;
            }
            if (v != null) {
                old.value = v;
            } else {
                removeNode(hash, key, null, false, true);
            }
            return v;
        } else {
            if (t != null) {
                t.putTreeVal(tab, hash, key, value);
            } else {
                tab[i] = newNode(hash, key, value, first);
                if (binCount >= TREEIFY_THRESHOLD - 1) {
                    treeifyBin(tab, hash);
                }
            }
            ++modCount;
            ++size;
            return value;
        }
    }

    public void forEach(BiConsumer<? super long[], ? super V> action) {
        Node<V>[] tab;
        if (action == null) {
            throw new NullPointerException();
        }
        if (size > 0 && (tab = table) != null) {
            int mc = modCount;
            for (Node<V> e : tab) {
                for (; e != null; e = e.next) {
                    action.accept(e.key, e.value);
                }
            }
            if (modCount != mc) {
                throw new ConcurrentModificationException();
            }
        }
    }

    public void replaceAll(BiFunction<? super long[], ? super V, ? extends V> function) {
        Node<V>[] tab;
        if (function == null) {
            throw new NullPointerException();
        }
        if (size > 0 && (tab = table) != null) {
            int mc = modCount;
            for (Node<V> e : tab) {
                for (; e != null; e = e.next) {
                    e.value = function.apply(e.key, e.value);
                }
            }
            if (modCount != mc) {
                throw new ConcurrentModificationException();
            }
        }
    }

    /* ------------------------------------------------------------ */
    // iterators

    private abstract class HashIterator {
        Node<V> next;        // next entry to return
        Node<V> current;     // current entry
        int expectedModCount;  // for fast-fail
        int index;             // current slot

        HashIterator() {
            expectedModCount = modCount;
            Node<V>[] t = table;
            current = next = null;
            index = 0;
            if (t != null && size > 0) { // advance to first entry
                do {
                } while (index < t.length && (next = t[index++]) == null);
            }
        }

        public final boolean hasNext() {
            return next != null;
        }

        final Node<V> nextNode() {
            Node<V>[] t;
            Node<V> e = next;
            if (modCount != expectedModCount) {
                throw new ConcurrentModificationException();
            }
            if (e == null) {
                throw new NoSuchElementException();
            }
            if ((next = (current = e).next) == null && (t = table) != null) {
                do {
                } while (index < t.length && (next = t[index++]) == null);
            }
            return e;
        }

        public final void remove() {
            Node<V> p = current;
            if (p == null) {
                throw new IllegalStateException();
            }
            if (modCount != expectedModCount) {
                throw new ConcurrentModificationException();
            }
            current = null;
            removeNode(p.hash, p.key, null, false, false);
            expectedModCount = modCount;
        }
    }

    private final class KeyIterator extends HashIterator
            implements Iterator<long[]> {
        public long[] next() {
            return nextNode().key;
        }
    }

    private final class ValueIterator extends HashIterator
            implements Iterator<V> {
        public V next() {
            return nextNode().value;
        }
    }

    private final class EntryIterator extends HashIterator
            implements Iterator<Node<V>> {
        public Node<V> next() {
            return nextNode();
        }
    }

    /* ------------------------------------------------------------ */
    // spliterators

    private static class HashMapSpliterator<V> {
        final LongArrayHashMap<V> map;
        Node<V> current;          // current node
        int index;                  // current index, modified on advance/split
        int fence;                  // one past last index
        int est;                    // size estimate
        int expectedModCount;       // for comodification checks

        HashMapSpliterator(LongArrayHashMap<V> m, int origin,
                int fence, int est,
                int expectedModCount) {
            this.map = m;
            this.index = origin;
            this.fence = fence;
            this.est = est;
            this.expectedModCount = expectedModCount;
        }

        final int getFence() { // initialize fence and size on first use
            int hi;
            if ((hi = fence) < 0) {
                LongArrayHashMap<V> m = map;
                est = m.size;
                expectedModCount = m.modCount;
                Node<V>[] tab = m.table;
                hi = fence = tab == null ? 0 : tab.length;
            }
            return hi;
        }

        public final long estimateSize() {
            getFence(); // force init
            return est;
        }
    }

    private static final class KeySpliterator<V>
            extends HashMapSpliterator<V>
            implements Spliterator<long[]> {
        KeySpliterator(LongArrayHashMap<V> m, int origin, int fence, int est,
                int expectedModCount) {
            super(m, origin, fence, est, expectedModCount);
        }

        public KeySpliterator<V> trySplit() {
            int hi = getFence(), lo = index, mid = lo + hi >>> 1;
            return lo >= mid || current != null ? null :
                    new KeySpliterator<>(map, lo, index = mid, est >>>= 1,
                            expectedModCount);
        }

        public void forEachRemaining(Consumer<? super long[]> action) {
            int i, hi, mc;
            if (action == null) {
                throw new NullPointerException();
            }
            LongArrayHashMap<V> m = map;
            Node<V>[] tab = m.table;
            if ((hi = fence) < 0) {
                mc = expectedModCount = m.modCount;
                hi = fence = tab == null ? 0 : tab.length;
            } else {
                mc = expectedModCount;
            }
            if (tab != null && tab.length >= hi &&
                (i = index) >= 0 && (i < (index = hi) || current != null)) {
                Node<V> p = current;
                current = null;
                do {
                    if (p == null) {
                        p = tab[i++];
                    } else {
                        action.accept(p.key);
                        p = p.next;
                    }
                } while (p != null || i < hi);
                if (m.modCount != mc) {
                    throw new ConcurrentModificationException();
                }
            }
        }

        public boolean tryAdvance(Consumer<? super long[]> action) {
            int hi;
            if (action == null) {
                throw new NullPointerException();
            }
            Node<V>[] tab = map.table;
            if (tab != null && tab.length >= (hi = getFence()) && index >= 0) {
                while (current != null || index < hi) {
                    if (current == null) {
                        current = tab[index++];
                    } else {
                        long[] k = current.key;
                        current = current.next;
                        action.accept(k);
                        if (map.modCount != expectedModCount) {
                            throw new ConcurrentModificationException();
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        public int characteristics() {
            return (fence < 0 || est == map.size ? Spliterator.SIZED : 0) |
                   Spliterator.DISTINCT;
        }
    }

    private static final class ValueSpliterator<V>
            extends HashMapSpliterator<V>
            implements Spliterator<V> {
        ValueSpliterator(LongArrayHashMap<V> m, int origin, int fence, int est,
                int expectedModCount) {
            super(m, origin, fence, est, expectedModCount);
        }

        public ValueSpliterator<V> trySplit() {
            int hi = getFence(), lo = index, mid = lo + hi >>> 1;
            return lo >= mid || current != null ? null :
                    new ValueSpliterator<>(map, lo, index = mid, est >>>= 1,
                            expectedModCount);
        }

        public void forEachRemaining(Consumer<? super V> action) {
            int i, hi, mc;
            if (action == null) {
                throw new NullPointerException();
            }
            LongArrayHashMap<V> m = map;
            Node<V>[] tab = m.table;
            if ((hi = fence) < 0) {
                mc = expectedModCount = m.modCount;
                hi = fence = tab == null ? 0 : tab.length;
            } else {
                mc = expectedModCount;
            }
            if (tab != null && tab.length >= hi &&
                (i = index) >= 0 && (i < (index = hi) || current != null)) {
                Node<V> p = current;
                current = null;
                do {
                    if (p == null) {
                        p = tab[i++];
                    } else {
                        action.accept(p.value);
                        p = p.next;
                    }
                } while (p != null || i < hi);
                if (m.modCount != mc) {
                    throw new ConcurrentModificationException();
                }
            }
        }

        public boolean tryAdvance(Consumer<? super V> action) {
            int hi;
            if (action == null) {
                throw new NullPointerException();
            }
            Node<V>[] tab = map.table;
            if (tab != null && tab.length >= (hi = getFence()) && index >= 0) {
                while (current != null || index < hi) {
                    if (current == null) {
                        current = tab[index++];
                    } else {
                        V v = current.value;
                        current = current.next;
                        action.accept(v);
                        if (map.modCount != expectedModCount) {
                            throw new ConcurrentModificationException();
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        public int characteristics() {
            return fence < 0 || est == map.size ? Spliterator.SIZED : 0;
        }
    }

    private static final class EntrySpliterator<V>
            extends HashMapSpliterator<V>
            implements Spliterator<Node<V>> {
        EntrySpliterator(LongArrayHashMap<V> m, int origin, int fence, int est,
                int expectedModCount) {
            super(m, origin, fence, est, expectedModCount);
        }

        public EntrySpliterator<V> trySplit() {
            int hi = getFence(), lo = index, mid = lo + hi >>> 1;
            return lo >= mid || current != null ? null :
                    new EntrySpliterator<>(map, lo, index = mid, est >>>= 1,
                            expectedModCount);
        }

        public void forEachRemaining(Consumer<? super Node<V>> action) {
            int i, hi, mc;
            if (action == null) {
                throw new NullPointerException();
            }
            LongArrayHashMap<V> m = map;
            Node<V>[] tab = m.table;
            if ((hi = fence) < 0) {
                mc = expectedModCount = m.modCount;
                hi = fence = tab == null ? 0 : tab.length;
            } else {
                mc = expectedModCount;
            }
            if (tab != null && tab.length >= hi &&
                (i = index) >= 0 && (i < (index = hi) || current != null)) {
                Node<V> p = current;
                current = null;
                do {
                    if (p == null) {
                        p = tab[i++];
                    } else {
                        action.accept(p);
                        p = p.next;
                    }
                } while (p != null || i < hi);
                if (m.modCount != mc) {
                    throw new ConcurrentModificationException();
                }
            }
        }

        public boolean tryAdvance(Consumer<? super Node<V>> action) {
            int hi;
            if (action == null) {
                throw new NullPointerException();
            }
            Node<V>[] tab = map.table;
            if (tab != null && tab.length >= (hi = getFence()) && index >= 0) {
                while (current != null || index < hi) {
                    if (current == null) {
                        current = tab[index++];
                    } else {
                        Node<V> e = current;
                        current = current.next;
                        action.accept(e);
                        if (map.modCount != expectedModCount) {
                            throw new ConcurrentModificationException();
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        public int characteristics() {
            return (fence < 0 || est == map.size ? Spliterator.SIZED : 0) |
                   Spliterator.DISTINCT;
        }
    }

    /* ------------------------------------------------------------ */
    // LinkedHashMap support


    /*
     * The following package-protected methods are designed to be
     * overridden by LinkedHashMap, but not by any other subclass.
     * Nearly all other internal methods are also package-protected
     * but are declared final, so can be used by LinkedHashMap, view
     * classes, and HashSet.
     */

    // Create a regular (non-tree) node
    private static <V> Node<V> newNode(int hash, long[] key, V value, Node<V> next) {
        return new Node<>(hash, key, value, next);
    }

    // For conversion from TreeNodes to plain nodes
    private static <V> Node<V> replacementNode(Node<V> p) {
        return new Node<>(p.hash, p.key, p.value, null);
    }

    // Create a tree bin node
    private static <V> TreeNode<V> newTreeNode(int hash, long[] key, V value, Node<V> next) {
        return new TreeNode<>(hash, key, value, next);
    }

    // For treeifyBin
    private static <V> TreeNode<V> replacementTreeNode(Node<V> p) {
        return new TreeNode<>(p.hash, p.key, p.value, null);
    }

    /* ------------------------------------------------------------ */
    // Tree bins

    @SuppressWarnings("ReplaceNullCheck")
    private static final class TreeNode<V> extends Node<V> {
        TreeNode<V> parent;  // red-black tree links
        TreeNode<V> left;
        TreeNode<V> right;
        TreeNode<V> prev;    // needed to unlink next upon deletion
        boolean red;

        TreeNode(int hash, long[] key, V val, Node<V> next) {
            super(hash, key, val, next);
        }

        /**
         * Returns root of tree containing this node.
         */
        TreeNode<V> root() {
            for (TreeNode<V> r = this, p; ; ) {
                if ((p = r.parent) == null) {
                    return r;
                }
                r = p;
            }
        }

        /**
         * Ensures that the given root is the first node of its bin.
         */
        static <V> void moveRootToFront(Node<V>[] tab, TreeNode<V> root) {
            int n;
            if (root != null && tab != null && (n = tab.length) > 0) {
                int index = n - 1 & root.hash;
                TreeNode<V> first = (TreeNode<V>) tab[index];
                if (root != first) {
                    Node<V> rn;
                    tab[index] = root;
                    TreeNode<V> rp = root.prev;
                    if ((rn = root.next) != null) {
                        ((TreeNode<V>) rn).prev = rp;
                    }
                    if (rp != null) {
                        rp.next = rn;
                    }
                    if (first != null) {
                        first.prev = root;
                    }
                    root.next = first;
                    root.prev = null;
                }
                assert checkInvariants(root);
            }
        }

        /**
         * Finds the node starting at root p with the given hash and key.
         * The kc argument caches comparableClassFor(key) upon first use
         * comparing keys.
         */
        TreeNode<V> find(int h, long[] k) {
            TreeNode<V> p = this;
            do {
                int ph, dir;
                long[] pk;
                TreeNode<V> pl = p.left, pr = p.right, q;
                if ((ph = p.hash) > h) {
                    p = pl;
                } else if (ph < h) {
                    p = pr;
                } else if (equal(k, pk = p.key)) {
                    return p;
                } else if (pl == null) {
                    p = pr;
                } else if (pr == null) {
                    p = pl;
                } else if ((dir = compareComparables(k, pk)) != 0) {
                    p = dir < 0 ? pl : pr;
                } else if ((q = pr.find(h, k)) != null) {
                    return q;
                } else {
                    p = pl;
                }
            } while (p != null);
            return null;
        }

        /**
         * Calls find for root node.
         */
        TreeNode<V> getTreeNode(int h, long[] k) {
            return (parent != null ? root() : this).find(h, k);
        }

        /**
         * Tie-breaking utility for ordering insertions when equal
         * hashCodes and non-comparable. We don't require a total
         * order, just a consistent insertion rule to maintain
         * equivalence across rebalancings. Tie-breaking further than
         * necessary simplifies testing a bit.
         */
        static int tieBreakOrder(long[] a, long[] b) {
            return System.identityHashCode(a) <= System.identityHashCode(b) ?
                    -1 : 1;
        }

        /**
         * Forms tree of the nodes linked from this node.
         */
        void treeify(Node<V>[] tab) {
            TreeNode<V> root = null;
            for (TreeNode<V> x = this, next; x != null; x = next) {
                next = (TreeNode<V>) x.next;
                x.left = x.right = null;
                if (root == null) {
                    x.parent = null;
                    x.red = false;
                    root = x;
                } else {
                    long[] k = x.key;
                    int h = x.hash;
                    for (TreeNode<V> p = root; ; ) {
                        int dir, ph;
                        long[] pk = p.key;
                        if ((ph = p.hash) > h) {
                            dir = -1;
                        } else if (ph < h) {
                            dir = 1;
                        } else if ((dir = compareComparables(k, pk)) == 0) {
                            dir = tieBreakOrder(k, pk);
                        }

                        TreeNode<V> xp = p;
                        if ((p = dir <= 0 ? p.left : p.right) == null) {
                            x.parent = xp;
                            if (dir <= 0) {
                                xp.left = x;
                            } else {
                                xp.right = x;
                            }
                            root = balanceInsertion(root, x);
                            break;
                        }
                    }
                }
            }
            moveRootToFront(tab, root);
        }

        /**
         * Returns a list of non-TreeNodes replacing those linked from
         * this node.
         */
        Node<V> untreeify() {
            Node<V> hd = null, tl = null;
            for (Node<V> q = this; q != null; q = q.next) {
                Node<V> p = replacementNode(q);
                if (tl == null) {
                    hd = p;
                } else {
                    tl.next = p;
                }
                tl = p;
            }
            return hd;
        }

        /**
         * Tree version of putVal.
         */
        TreeNode<V> putTreeVal(Node<V>[] tab,
                int h, long[] k, V v) {
            boolean searched = false;
            TreeNode<V> root = parent != null ? root() : this;
            for (TreeNode<V> p = root; ; ) {
                int dir, ph;
                long[] pk;
                if ((ph = p.hash) > h) {
                    dir = -1;
                } else if (ph < h) {
                    dir = 1;
                } else if (equal(k, pk = p.key)) {
                    return p;
                } else if ((dir = compareComparables(k, pk)) == 0) {
                    if (!searched) {
                        TreeNode<V> q, ch;
                        searched = true;
                        if ((ch = p.left) != null &&
                            (q = ch.find(h, k)) != null ||
                            (ch = p.right) != null &&
                            (q = ch.find(h, k)) != null) {
                            return q;
                        }
                    }
                    dir = tieBreakOrder(k, pk);
                }

                TreeNode<V> xp = p;
                if ((p = dir <= 0 ? p.left : p.right) == null) {
                    Node<V> xpn = xp.next;
                    TreeNode<V> x = newTreeNode(h, k, v, xpn);
                    if (dir <= 0) {
                        xp.left = x;
                    } else {
                        xp.right = x;
                    }
                    xp.next = x;
                    x.parent = x.prev = xp;
                    if (xpn != null) {
                        ((TreeNode<V>) xpn).prev = x;
                    }
                    moveRootToFront(tab, balanceInsertion(root, x));
                    return null;
                }
            }
        }

        /**
         * Removes the given node, that must be present before this call.
         * This is messier than typical red-black deletion code because we
         * cannot swap the contents of an interior node with a leaf
         * successor that is pinned by "next" pointers that are accessible
         * independently during traversal. So instead we swap the tree
         * linkages. If the current tree appears to have too few nodes,
         * the bin is converted back to a plain bin. (The test triggers
         * somewhere between 2 and 6 nodes, depending on tree structure).
         */
        void removeTreeNode(Node<V>[] tab,
                boolean movable) {
            int n;
            if (tab == null || (n = tab.length) == 0) {
                return;
            }
            int index = n - 1 & hash;
            TreeNode<V> first = (TreeNode<V>) tab[index], root = first, rl;
            TreeNode<V> succ = (TreeNode<V>) next, pred = prev;
            if (pred == null) {
                tab[index] = first = succ;
            } else {
                pred.next = succ;
            }
            if (succ != null) {
                succ.prev = pred;
            }
            if (first == null) {
                return;
            }
            if (root.parent != null) {
                root = root.root();
            }
            if (movable && (root.right == null || (rl = root.left) == null || rl.left == null)) {
                tab[index] = first.untreeify();  // too small
                return;
            }
            TreeNode<V> p = this, pl = left, pr = right, replacement;
            if (pl != null && pr != null) {
                TreeNode<V> s = pr, sl;
                while ((sl = s.left) != null) // find successor
                {
                    s = sl;
                }
                boolean c = s.red;
                s.red = p.red;
                p.red = c; // swap colors
                TreeNode<V> sr = s.right;
                TreeNode<V> pp = p.parent;
                if (s == pr) { // p was s's direct parent
                    p.parent = s;
                    s.right = p;
                } else {
                    TreeNode<V> sp = s.parent;
                    if ((p.parent = sp) != null) {
                        if (s == sp.left) {
                            sp.left = p;
                        } else {
                            sp.right = p;
                        }
                    }
                    s.right = pr;
                    pr.parent = s;
                }
                p.left = null;
                if ((p.right = sr) != null) {
                    sr.parent = p;
                }
                s.left = pl;
                pl.parent = s;
                if ((s.parent = pp) == null) {
                    root = s;
                } else if (p == pp.left) {
                    pp.left = s;
                } else {
                    pp.right = s;
                }
                if (sr != null) {
                    replacement = sr;
                } else {
                    replacement = p;
                }
            } else if (pl != null) {
                replacement = pl;
            } else if (pr != null) {
                replacement = pr;
            } else {
                replacement = p;
            }
            if (replacement != p) {
                TreeNode<V> pp = replacement.parent = p.parent;
                if (pp == null) {
                    (root = replacement).red = false;
                } else if (p == pp.left) {
                    pp.left = replacement;
                } else {
                    pp.right = replacement;
                }
                p.left = p.right = p.parent = null;
            }

            TreeNode<V> r = p.red ? root : balanceDeletion(root, replacement);

            if (replacement == p) {  // detach
                TreeNode<V> pp = p.parent;
                p.parent = null;
                if (pp != null) {
                    if (p == pp.left) {
                        pp.left = null;
                    } else if (p == pp.right) {
                        pp.right = null;
                    }
                }
            }
            if (movable) {
                moveRootToFront(tab, r);
            }
        }

        /**
         * Splits nodes in a tree bin into lower and upper tree bins,
         * or untreeifies if now too small. Called only from resize;
         * see above discussion about split bits and indices.
         *
         * @param tab the table for recording bin heads
         * @param index the index of the table being split
         * @param bit the bit of hash to split on
         */
        void split(Node<V>[] tab, int index, int bit) {
            TreeNode<V> b = this;
            // Relink into lo and hi lists, preserving order
            TreeNode<V> loHead = null, loTail = null;
            TreeNode<V> hiHead = null, hiTail = null;
            int lc = 0, hc = 0;
            for (TreeNode<V> e = b, next; e != null; e = next) {
                next = (TreeNode<V>) e.next;
                e.next = null;
                if ((e.hash & bit) == 0) {
                    if ((e.prev = loTail) == null) {
                        loHead = e;
                    } else {
                        loTail.next = e;
                    }
                    loTail = e;
                    ++lc;
                } else {
                    if ((e.prev = hiTail) == null) {
                        hiHead = e;
                    } else {
                        hiTail.next = e;
                    }
                    hiTail = e;
                    ++hc;
                }
            }

            if (loHead != null) {
                if (lc <= UNTREEIFY_THRESHOLD) {
                    tab[index] = loHead.untreeify();
                } else {
                    tab[index] = loHead;
                    if (hiHead != null) // (else is already treeified)
                    {
                        loHead.treeify(tab);
                    }
                }
            }
            if (hiHead != null) {
                if (hc <= UNTREEIFY_THRESHOLD) {
                    tab[index + bit] = hiHead.untreeify();
                } else {
                    tab[index + bit] = hiHead;
                    if (loHead != null) {
                        hiHead.treeify(tab);
                    }
                }
            }
        }

        /* ------------------------------------------------------------ */
        // Red-black tree methods, all adapted from CLR

        static <V> TreeNode<V> rotateLeft(TreeNode<V> root,
                TreeNode<V> p) {
            TreeNode<V> r, pp, rl;
            if (p != null && (r = p.right) != null) {
                if ((rl = p.right = r.left) != null) {
                    rl.parent = p;
                }
                if ((pp = r.parent = p.parent) == null) {
                    (root = r).red = false;
                } else if (pp.left == p) {
                    pp.left = r;
                } else {
                    pp.right = r;
                }
                r.left = p;
                p.parent = r;
            }
            return root;
        }

        static <V> TreeNode<V> rotateRight(TreeNode<V> root,
                TreeNode<V> p) {
            TreeNode<V> l, pp, lr;
            if (p != null && (l = p.left) != null) {
                if ((lr = p.left = l.right) != null) {
                    lr.parent = p;
                }
                if ((pp = l.parent = p.parent) == null) {
                    (root = l).red = false;
                } else if (pp.right == p) {
                    pp.right = l;
                } else {
                    pp.left = l;
                }
                l.right = p;
                p.parent = l;
            }
            return root;
        }

        static <V> TreeNode<V> balanceInsertion(TreeNode<V> root,
                TreeNode<V> x) {
            x.red = true;
            for (TreeNode<V> xp, xpp, xppl, xppr; ; ) {
                if ((xp = x.parent) == null) {
                    x.red = false;
                    return x;
                } else if (!xp.red || (xpp = xp.parent) == null) {
                    return root;
                }
                if (xp == (xppl = xpp.left)) {
                    if ((xppr = xpp.right) != null && xppr.red) {
                        xppr.red = false;
                        xp.red = false;
                        xpp.red = true;
                        x = xpp;
                    } else {
                        if (x == xp.right) {
                            root = rotateLeft(root, x = xp);
                            xpp = (xp = x.parent) == null ? null : xp.parent;
                        }
                        if (xp != null) {
                            xp.red = false;
                            if (xpp != null) {
                                xpp.red = true;
                                root = rotateRight(root, xpp);
                            }
                        }
                    }
                } else {
                    if (xppl != null && xppl.red) {
                        xppl.red = false;
                        xp.red = false;
                        xpp.red = true;
                        x = xpp;
                    } else {
                        if (x == xp.left) {
                            root = rotateRight(root, x = xp);
                            xpp = (xp = x.parent) == null ? null : xp.parent;
                        }
                        if (xp != null) {
                            xp.red = false;
                            if (xpp != null) {
                                xpp.red = true;
                                root = rotateLeft(root, xpp);
                            }
                        }
                    }
                }
            }
        }

        static <V> TreeNode<V> balanceDeletion(TreeNode<V> root,
                TreeNode<V> x) {
            for (TreeNode<V> xp, xpl, xpr; ; ) {
                if (x == null || x == root) {
                    return root;
                } else if ((xp = x.parent) == null) {
                    x.red = false;
                    return x;
                } else if (x.red) {
                    x.red = false;
                    return root;
                } else if ((xpl = xp.left) == x) {
                    if ((xpr = xp.right) != null && xpr.red) {
                        xpr.red = false;
                        xp.red = true;
                        root = rotateLeft(root, xp);
                        xpr = (xp = x.parent) == null ? null : xp.right;
                    }
                    if (xpr == null) {
                        x = xp;
                    } else {
                        TreeNode<V> sl = xpr.left, sr = xpr.right;
                        if ((sr == null || !sr.red) &&
                            (sl == null || !sl.red)) {
                            xpr.red = true;
                            x = xp;
                        } else {
                            if (sr == null || !sr.red) {
                                sl.red = false;
                                xpr.red = true;
                                root = rotateRight(root, xpr);
                                xpr = (xp = x.parent) == null ?
                                        null : xp.right;
                            }
                            if (xpr != null) {
                                xpr.red = xp.red;
                                if ((sr = xpr.right) != null) {
                                    sr.red = false;
                                }
                            }
                            if (xp != null) {
                                xp.red = false;
                                root = rotateLeft(root, xp);
                            }
                            x = root;
                        }
                    }
                } else { // symmetric
                    if (xpl != null && xpl.red) {
                        xpl.red = false;
                        xp.red = true;
                        root = rotateRight(root, xp);
                        xpl = (xp = x.parent) == null ? null : xp.left;
                    }
                    if (xpl == null) {
                        x = xp;
                    } else {
                        TreeNode<V> sl = xpl.left, sr = xpl.right;
                        if ((sl == null || !sl.red) &&
                            (sr == null || !sr.red)) {
                            xpl.red = true;
                            x = xp;
                        } else {
                            if (sl == null || !sl.red) {
                                sr.red = false;
                                xpl.red = true;
                                root = rotateLeft(root, xpl);
                                xpl = (xp = x.parent) == null ?
                                        null : xp.left;
                            }
                            if (xpl != null) {
                                xpl.red = xp.red;
                                if ((sl = xpl.left) != null) {
                                    sl.red = false;
                                }
                            }
                            if (xp != null) {
                                xp.red = false;
                                root = rotateRight(root, xp);
                            }
                            x = root;
                        }
                    }
                }
            }
        }

        /**
         * Recursive invariant check
         */
        static <V> boolean checkInvariants(TreeNode<V> t) {
            TreeNode<V> tp = t.parent, tl = t.left, tr = t.right,
                    tb = t.prev, tn = (TreeNode<V>) t.next;
            if (tb != null && tb.next != t) {
                return false;
            }
            if (tn != null && tn.prev != t) {
                return false;
            }
            if (tp != null && t != tp.left && t != tp.right) {
                return false;
            }
            if (tl != null && (tl.parent != t || tl.hash > t.hash)) {
                return false;
            }
            if (tr != null && (tr.parent != t || tr.hash < t.hash)) {
                return false;
            }
            if (t.red && tl != null && tl.red && tr != null && tr.red) {
                return false;
            }
            if (tl != null && !checkInvariants(tl)) {
                return false;
            }
            if (tr != null && !checkInvariants(tr)) {
                return false;
            }
            return true;
        }
    }

    /**
     * Calculate initial capacity for HashMap based classes, from expected size and default load factor (0.75).
     *
     * @param numMappings the expected number of mappings
     * @return initial capacity for HashMap based classes.
     * @since 19
     */
    private static int calculateHashMapCapacity(int numMappings) {
        return (int) Math.ceil(numMappings / (double) DEFAULT_LOAD_FACTOR);
    }

    /**
     * Creates a new, empty HashMap suitable for the expected number of mappings.
     * The returned map uses the default load factor of 0.75, and its initial capacity is
     * generally large enough so that the expected number of mappings can be added
     * without resizing the map.
     *
     * @param numMappings the expected number of mappings
     * @param <V> the type of mapped values
     * @return the newly created map
     * @throws IllegalArgumentException if numMappings is negative
     * @since 19
     */
    public static <V> LongArrayHashMap<V> newHashMap(int numMappings) {
        if (numMappings < 0) {
            throw new IllegalArgumentException("Negative number of mappings: " + numMappings);
        }
        return new LongArrayHashMap<>(calculateHashMapCapacity(numMappings));
    }

}
