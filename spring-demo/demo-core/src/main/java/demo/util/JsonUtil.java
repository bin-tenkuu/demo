package demo.util;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.fasterxml.jackson.datatype.jsr310.deser.LocalDateDeserializer;
import com.fasterxml.jackson.datatype.jsr310.deser.LocalDateTimeDeserializer;
import com.fasterxml.jackson.datatype.jsr310.deser.LocalTimeDeserializer;
import com.fasterxml.jackson.datatype.jsr310.ser.LocalDateSerializer;
import com.fasterxml.jackson.datatype.jsr310.ser.LocalDateTimeSerializer;
import com.fasterxml.jackson.datatype.jsr310.ser.LocalTimeSerializer;
import org.intellij.lang.annotations.Language;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import java.util.Map;

import static demo.constant.DateConstant.*;

/**
 * @author bin
 * @since 2023/05/30
 */
@SuppressWarnings("unused")
public class JsonUtil {

    @FunctionalInterface
    public interface Func1<T, R> {
        R run(T t) throws Exception;
    }

    @FunctionalInterface
    public interface Func2<T1, T2, R> {
        R run(T1 t1, T2 t2) throws Exception;
    }

    @FunctionalInterface
    public interface Call2<T1, T2> {
        void call(T1 t, T2 t2) throws Exception;
    }

    public static final ObjectMapper objectMapper = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
    };

    static {
        var sampleModel = new SimpleModule();
        sampleModel.addSerializer(Long.class, ToStringSerializer.instance)
                .addSerializer(Long.TYPE, ToStringSerializer.instance)
                .addSerializer(LocalDateTime.class, new LocalDateTimeSerializer(DATE_TIME_FORMATTER))
                .addDeserializer(LocalDateTime.class, new LocalDateTimeDeserializer(DATE_TIME_FORMATTER))
                .addSerializer(LocalDate.class, new LocalDateSerializer(DATE_FORMATTER))
                .addDeserializer(LocalDate.class, new LocalDateDeserializer(DATE_FORMATTER))
                .addSerializer(LocalTime.class, new LocalTimeSerializer(TIME_FORMATTER))
                .addDeserializer(LocalTime.class, new LocalTimeDeserializer(TIME_FORMATTER));
        objectMapper
                .configure(DeserializationFeature.ADJUST_DATES_TO_CONTEXT_TIME_ZONE, false)
                .configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false)
                .configure(JsonGenerator.Feature.WRITE_BIGDECIMAL_AS_PLAIN, true)
                .registerModules(new JavaTimeModule())
                .registerModule(sampleModel);
    }

    @Bean
    public ObjectMapper objectMapper() {
        return objectMapper;
    }

    public static <T, R> R tryParse(T t, Func1<T, R> callable) {
        try {
            return callable.run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T1, T2, R> R tryParse(T1 t1, T2 t2, Func2<T1, T2, R> callable) {
        try {
            return callable.run(t1, t2);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T1, T2> void tryCall(T1 t1, T2 t2, Call2<T1, T2> callable) {
        try {
            callable.call(t1, t2);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Autowired
    public void setJackson2ObjectMapperBuilder(Jackson2ObjectMapperBuilder builder) {
        builder.createXmlMapper(false).configure(objectMapper);
    }

    public static <T> String toJson(T value) {
        if (value == null) {
            return "";
        }
        if (value instanceof CharSequence) {
            return value.toString();
        }
        return tryParse(value, objectMapper::writeValueAsString);
    }

    public static <T> byte[] toJsonBytes(T value) {
        if (value == null) {
            return new byte[0];
        }
        if (value instanceof CharSequence) {
            return value.toString().getBytes(StandardCharsets.UTF_8);
        }
        return tryParse(value, objectMapper::writeValueAsBytes);
    }

    public static <T> void toJsonTo(T value, File file) {
        tryCall(file, value, objectMapper::writeValue);
    }

    public static <T> void toJsonTo(T value, OutputStream out) {
        tryCall(out, value, objectMapper::writeValue);
    }

    public static <T> void toJsonTo(T value, DataOutput out) {
        tryCall(out, value, objectMapper::writeValue);
    }

    public static <T> void toJsonTo(T value, Writer w) {
        tryCall(w, value, objectMapper::writeValue);
    }

    public static Map<String, Object> toMap(@Language("json") String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        return tryParse(json, MAP_TYPE, objectMapper::readValue);
    }

    public static JsonNode toBean(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        return tryParse(json, objectMapper::readTree);
    }

    @SuppressWarnings("unchecked")
    public static <T> T toBean(String json, Class<T> clazz) {
        if (json == null || json.isBlank()) {
            return null;
        }
        if (String.class.isAssignableFrom(clazz)) {
            return (T) json;
        }
        return tryParse(json, clazz, objectMapper::readValue);
    }

    public static <T> T toBean(String json, TypeReference<T> clazz) {
        if (json == null || json.isBlank()) {
            return null;
        }
        return tryParse(json, clazz, objectMapper::readValue);
    }

    public static <T> T toBean(String json, JavaType clazz) {
        if (json == null || json.isBlank()) {
            return null;
        }
        return tryParse(json, clazz, objectMapper::readValue);
    }

    @SuppressWarnings("unchecked")
    public static <T> T toBean(byte[] json, Class<T> clazz) {
        if (json == null || json.length == 0) {
            return null;
        }
        if (String.class.isAssignableFrom(clazz)) {
            return (T) new String(json);
        }
        return tryParse(json, clazz, objectMapper::readValue);
    }

    public static <T> T toBean(byte[] json, TypeReference<T> clazz) {
        if (json == null || json.length == 0) {
            return null;
        }
        return tryParse(json, clazz, objectMapper::readValue);
    }

    public static <T> T toBean(byte[] json, JavaType clazz) {
        if (json == null || json.length == 0) {
            return null;
        }
        return tryParse(json, clazz, objectMapper::readValue);
    }

    public static <T> List<T> toBeanList(String json, Class<T> clazz) {
        var type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> List<T> toBeanList(String json, JavaType clazz) {
        var type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> List<T> toBeanList(byte[] json, Class<T> clazz) {
        var type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> List<T> toBeanList(byte[] json, JavaType clazz) {
        var type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> T convertBean(Object obj, Class<T> clazz) {
        if (obj == null) {
            return null;
        }
        return tryParse(obj, clazz, objectMapper::convertValue);
    }

    public static <T> T convertBean(Object obj, TypeReference<T> clazz) {
        if (obj == null) {
            return null;
        }
        return tryParse(obj, clazz, objectMapper::convertValue);
    }

    public static <T> T convertBean(Object obj, JavaType clazz) {
        if (obj == null) {
            return null;
        }
        return tryParse(obj, clazz, objectMapper::convertValue);
    }

    public static Map<String, Object> convertMap(Object obj) {
        return convertBean(obj, MAP_TYPE);
    }

    // region toParser
    public static JsonParser toParser(String json) {
        return tryParse(json, objectMapper::createParser);
    }

    public static JsonParser toParser(byte[] json) {
        return tryParse(json, objectMapper::createParser);
    }

    public static JsonParser toParser(File json) {
        return tryParse(json, objectMapper::createParser);
    }

    public static JsonParser toParser(InputStream json) {
        return tryParse(json, objectMapper::createParser);
    }

    public static JsonParser toParser(Reader json) {
        return tryParse(json, objectMapper::createParser);
    }

    // endregion
    // region toJavaType
    public static JavaType getJavaType(Type type) {
        return objectMapper.getTypeFactory().constructType(type);
    }

    public static JavaType getListJavaType(Class<?> type) {
        return objectMapper.getTypeFactory().constructCollectionType(List.class, type);
    }

    public static JavaType getMapJavaType(Class<?> key, Class<?> value) {
        return objectMapper.getTypeFactory().constructMapType(Map.class, key, value);
    }

    public static <T> JavaType getJavaType(TypeReference<T> type) {
        return objectMapper.getTypeFactory().constructType(type);
    }
    // endregion
}
