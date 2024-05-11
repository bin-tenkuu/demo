package demo.test;

import java.util.*;
import java.util.function.*;

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
public final class LongHashMap<V> {

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

    public static class Node<V> {
        final int hash;
        final long key;
        V value;
        Node<V> next;

        Node(int hash, long key, V value, Node<V> next) {
            this.hash = hash;
            this.key = key;
            this.value = value;
            this.next = next;
        }

        public final long getKey() {
            return key;
        }

        public final V getValue() {
            return value;
        }

        public final String toString() {
            return key + "=" + value;
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
                   && key == e.getKey()
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
    private static int hash(long key) {
        return Long.hashCode(key);
    }

    private static boolean equal(long k, long x) {
        return k == x;
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
     */
    public LongHashMap(int initialCapacity, float loadFactor) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Illegal initial capacity: " +
                                               initialCapacity);
        }
        if (initialCapacity > MAXIMUM_CAPACITY) {
            initialCapacity = MAXIMUM_CAPACITY;
        }
        if (loadFactor <= 0 || Float.isNaN(loadFactor)) {
            throw new IllegalArgumentException("Illegal load factor: " +
                                               loadFactor);
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
     */
    public LongHashMap(int initialCapacity) {
        this(initialCapacity, DEFAULT_LOAD_FACTOR);
    }

    /**
     * Constructs an empty {@code HashMap} with the default initial capacity
     * (16) and the default load factor (0.75).
     */
    public LongHashMap() {
        this.loadFactor = DEFAULT_LOAD_FACTOR; // all other fields defaulted
    }

    /**
     * Implements Map.putAll and Map constructor.
     *
     * @param m the map
     */
    private void putMapEntries(LongHashMap<? extends V> m) {
        int s = m.size();
        if (s > 0) {
            if (table == null) { // pre-size
                double dt = Math.ceil(s / (double) loadFactor);
                int t = dt < (double) MAXIMUM_CAPACITY ?
                        (int) dt : MAXIMUM_CAPACITY;
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

            for (var e : m.entrySet()) {
                var key = e.getKey();
                var value = e.getValue();
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
     * @see #put(long, Object)
     */
    public V get(long key) {
        Node<V> e;
        return (e = getNode(key)) == null ? null : e.value;
    }

    /**
     * Implements Map.get and related methods.
     *
     * @param key the key
     * @return the node, or null if none
     */
    private Node<V> getNode(long key) {
        Node<V>[] tab;
        Node<V> first, e;
        int n, hash;
        if ((tab = table) != null && (n = tab.length) > 0 &&
            (first = tab[n - 1 & (hash = hash(key))]) != null) {
            if (first.hash == hash && equal(key, first.key)) {
                return first;
            }
            for (e = first.next; e != null; e = e.next) {
                if (e.hash == hash && equal(key, e.key)) {
                    return e;
                }
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
    public boolean containsKey(long key) {
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
    public V put(long key, V value) {
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
    private V putVal(int hash, long key, V value, boolean onlyIfAbsent) {
        Node<V>[] tab;
        Node<V> p;
        int n, i;
        if ((tab = table) == null || (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        if ((p = tab[i = n - 1 & hash]) == null) {
            tab[i] = newNode(hash, key, value, null);
        } else {
            Node<V> e;
            if (p.hash == hash &&
                equal(key, p.key)) {
                e = p;
            } else {
                while (true) {
                    e = p.next;
                    if (e == null) {
                        p.next = newNode(hash, key, value, null);
                        break;
                    }
                    if (e.hash == hash && equal(key, e.key)) {
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
     * Copies all of the mappings from the specified map to this map.
     * These mappings will replace any mappings that this map had for
     * any of the keys currently in the specified map.
     *
     * @param m mappings to be stored in this map
     * @throws NullPointerException if the specified map is null
     */
    public void putAll(LongHashMap<? extends V> m) {
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
    public V remove(long key) {
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
    private Node<V> removeNode(int hash, long key, Object value,
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
                do {
                    if (e.hash == hash && equal(key, e.key)) {
                        node = e;
                        break;
                    }
                    p = e;
                } while ((e = e.next) != null);
            }
            if (node != null && (!matchValue || (v = node.value) == value ||
                                 value != null && value.equals(v))) {
                if (node == p) {
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
    private long[] keysToArray() {
        long[] r = new long[size];
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
            LongHashMap.this.clear();
        }

        public Iterator<V> iterator() {
            return new ValueIterator();
        }

        public boolean contains(Object o) {
            return containsValue(o);
        }

        public Spliterator<V> spliterator() {
            return new ValueSpliterator<>(LongHashMap.this, 0, -1, 0, 0);
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
            LongHashMap.this.clear();
        }

        public Iterator<Node<V>> iterator() {
            return new EntryIterator();
        }

        public boolean contains(Object o) {
            if (!(o instanceof Node<?> e)) {
                return false;
            }
            long key = e.getKey();
            Node<V> candidate = getNode(key);
            return candidate != null && candidate.equals(e);
        }

        public boolean remove(Object o) {
            if (o instanceof Node<?> e) {
                long key = e.getKey();
                Object value = e.getValue();
                return removeNode(hash(key), key, value, true, true) != null;
            }
            return false;
        }

        public Spliterator<Node<V>> spliterator() {
            return new EntrySpliterator<>(LongHashMap.this, 0, -1, 0, 0);
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

    public V getOrDefault(long key, V defaultValue) {
        Node<V> e;
        return (e = getNode(key)) == null ? defaultValue : e.value;
    }

    public V putIfAbsent(long key, V value) {
        return putVal(hash(key), key, value, true);
    }

    public boolean remove(long key, Object value) {
        return removeNode(hash(key), key, value, true, true) != null;
    }

    public boolean replace(long key, V oldValue, V newValue) {
        Node<V> e;
        V v;
        if ((e = getNode(key)) != null &&
            ((v = e.value) == oldValue || v != null && v.equals(oldValue))) {
            e.value = newValue;
            return true;
        }
        return false;
    }

    public V replace(long key, V value) {
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
    public V computeIfAbsent(long key,
            LongFunction<? extends V> mappingFunction) {
        if (mappingFunction == null) {
            throw new NullPointerException();
        }
        int hash = hash(key);
        Node<V>[] tab;
        Node<V> first;
        int n, i;
        Node<V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        if ((first = tab[i = n - 1 & hash]) != null) {
            Node<V> e = first;
            do {
                if (e.hash == hash &&
                    equal(key, e.key)) {
                    old = e;
                    break;
                }
            } while ((e = e.next) != null);
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
        } else {
            tab[i] = newNode(hash, key, v, first);
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
    public V merge(long key, V value,
            BiFunction<? super V, ? super V, ? extends V> remappingFunction) {
        if (value == null || remappingFunction == null) {
            throw new NullPointerException();
        }
        int hash = hash(key);
        Node<V>[] tab;
        Node<V> first;
        int n, i;
        Node<V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        if ((first = tab[i = n - 1 & hash]) != null) {
            Node<V> e = first;
            do {
                if (e.hash == hash &&
                    equal(key, e.key)) {
                    old = e;
                    break;
                }
            } while ((e = e.next) != null);
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
            tab[i] = newNode(hash, key, value, first);
            ++modCount;
            ++size;
            return value;
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
        final LongHashMap<V> map;
        Node<V> current;          // current node
        int index;                  // current index, modified on advance/split
        int fence;                  // one past last index
        int est;                    // size estimate
        int expectedModCount;       // for comodification checks

        HashMapSpliterator(LongHashMap<V> m, int origin,
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
                LongHashMap<V> m = map;
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

    private static final class ValueSpliterator<V>
            extends HashMapSpliterator<V>
            implements Spliterator<V> {
        ValueSpliterator(LongHashMap<V> m, int origin, int fence, int est,
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
            LongHashMap<V> m = map;
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
        EntrySpliterator(LongHashMap<V> m, int origin, int fence, int est,
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
            LongHashMap<V> m = map;
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

    // Create a regular (non-tree) node
    private static <V> Node<V> newNode(int hash, long key, V value, Node<V> next) {
        return new Node<>(hash, key, value, next);
    }

}
