package demo.core.config;

import lombok.val;
import org.jspecify.annotations.NonNull;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.connection.MessageListener;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.*;
import org.springframework.data.redis.listener.ChannelTopic;
import org.springframework.data.redis.listener.RedisMessageListenerContainer;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.data.redis.serializer.SerializationException;
import org.springframework.data.redis.serializer.StringRedisSerializer;
import org.springframework.stereotype.Service;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.JavaType;
import tools.jackson.databind.json.JsonMapper;

import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author bin
 * @since 2024/05/11
 */
@SuppressWarnings("unused")
@Service
public class RawRedisTemplate extends RedisTemplate<String, Object> {
    private static final JsonMapper json = new JsonMapper();
    private static final byte[] EMPTY_ARRAY = new byte[0];
    private static final JavaType DEFAULT = getJavaType(byte[].class);
    private static final ThreadLocal<JavaType> JAVA_TYPE = ThreadLocal.withInitial(() -> DEFAULT);
    private static final StringRedisSerializer KEY_SER = StringRedisSerializer.UTF_8;
    private static final RedisSerializer<Object> VALUE_SER = new RedisSerializer<>() {
        @Override
        public byte @NonNull [] serialize(Object obj) throws SerializationException {
            return switch (obj) {
                case null -> EMPTY_ARRAY;
                case byte[] bytes -> bytes;
                case String str -> KEY_SER.serialize(str);
                default -> json.writeValueAsBytes(obj);
            };
        }

        @Override
        public Object deserialize(byte[] bytes) throws SerializationException {
            var clazz = JAVA_TYPE.get();
            if (bytes == null) {
                return null;
            } else if (clazz.isTypeOrSubTypeOf(String.class)) {
                return KEY_SER.deserialize(bytes);
            } else if (clazz.isTypeOrSubTypeOf(byte[].class)) {
                return bytes;
            }
            return json.readValue(bytes, clazz);
        }
    };

    // region field
    public final RedisMessageListenerContainer container;

    public RawRedisTemplate(RedisConnectionFactory connectionFactory) {
        setConnectionFactory(connectionFactory);
        setStringSerializer(KEY_SER);
        setDefaultSerializer(KEY_SER);
        setKeySerializer(KEY_SER);
        setValueSerializer(VALUE_SER);
        setHashKeySerializer(KEY_SER);
        setHashValueSerializer(VALUE_SER);
        container = new RedisMessageListenerContainer();
        container.setConnectionFactory(connectionFactory);
    }

    // @Bean
    public RedisMessageListenerContainer messageListenerContainer() {
        return container;
    }

    // endregion
    // region Serialize

    private static JavaType getJavaType(Type type) {
        return json.getTypeFactory().constructType(type);
    }

    private static JavaType getJavaType(TypeReference<?> type) {
        return json.getTypeFactory().constructType(type);
    }

    private static void setType(Type type) {
        JAVA_TYPE.set(getJavaType(type));
    }

    private static void setType(TypeReference<?> type) {
        JAVA_TYPE.set(getJavaType(type));
    }

    // endregion
    // region Operations

    @SuppressWarnings("unchecked")
    public <V> ClusterOperations<String, V> forCluster(Class<V> clazz) {
        setType(clazz);
        return (ClusterOperations<String, V>) opsForCluster();
    }

    @SuppressWarnings("unchecked")
    public <V> GeoOperations<String, V> forGeo(Class<V> clazz) {
        setType(clazz);
        return (GeoOperations<String, V>) opsForGeo();
    }

    public <V> HashOperations<String, String, V> forHash(Class<V> clazz) {
        setType(clazz);
        return opsForHash();
    }

    @SuppressWarnings("unchecked")
    public <V> ListOperations<String, V> forList(Class<V> clazz) {
        setType(clazz);
        return (ListOperations<String, V>) opsForList();
    }

    @SuppressWarnings("unchecked")
    public <V> SetOperations<String, V> forSet(Class<V> clazz) {
        setType(clazz);
        return (SetOperations<String, V>) opsForSet();
    }

    @SuppressWarnings("unchecked")
    public <V> ValueOperations<String, V> forValue(Class<V> clazz) {
        setType(clazz);
        return (ValueOperations<String, V>) opsForValue();
    }

    @SuppressWarnings("unchecked")
    public <V> ZSetOperations<String, V> forZSet(Class<V> clazz) {
        setType(clazz);
        return (ZSetOperations<String, V>) opsForZSet();
    }

    /// 君子协定方法
    /// session 方法运行期间调用的所有 redis 操作返回值都延迟到 pipeline 执行完成后统一返回
    /// 只要保证所有返回值都是这个类，就可以少写一个强转
    /// 批量操作中使用 [#executePipelined(Class, Callback)] 可以减少延迟、负载、时间，保证操作原子性
    @SuppressWarnings("unchecked")
    public <T> List<T> executePipelined(Class<T> clazz, Callback<T> session) {
        setType(clazz);
        return (List<T>) executePipelined(new SessionCallback<>() {
            @Override
            public <K, V> Object execute(@NonNull RedisOperations<K, V> o) throws DataAccessException {
                session.run();
                return null;
            }
        }, VALUE_SER);
    }

    @FunctionalInterface
    public interface Callback<T> extends Runnable {
    }

    // endregion
    // region subscribe
    public void publish(String channel, Object message) {
        convertAndSend(channel, VALUE_SER.serialize(message));
    }

    public void subscribe(MessageListener listener, String... channelName) {
        val list = Arrays.stream(channelName)
                .map(ChannelTopic::new)
                .collect(Collectors.toList());
        container.addMessageListener(listener, list);
    }

    public void subscribe(MessageListener listener, Collection<String> channelName) {
        val list = channelName.stream()
                .map(ChannelTopic::new)
                .collect(Collectors.toList());
        container.addMessageListener(listener, list);
    }

    public void unsubscribe(MessageListener listener) {
        container.removeMessageListener(listener);
    }
    // endregion

}
