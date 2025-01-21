package demo.util;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
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
import lombok.val;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder;

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
    public static final ObjectMapper objectMapper = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
    };

    static {
        val sampleModel = new SimpleModule();
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

    @Bean()
    public ObjectMapper objectMapper() {
        return objectMapper;
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
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> byte[] toJsonBytes(T value) {
        if (value == null) {
            return new byte[0];
        }
        if (value instanceof CharSequence) {
            return value.toString().getBytes(StandardCharsets.UTF_8);
        }
        try {
            return objectMapper.writeValueAsBytes(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static Map<String, Object> toMap(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, MAP_TYPE);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static JsonNode toBean(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readTree(json);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T toBean(String json, Class<T> clazz) {
        if (json == null || json.isBlank()) {
            return null;
        }
        if (String.class.isAssignableFrom(clazz)) {
            return (T) json;
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T toBean(String json, TypeReference<T> clazz) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T toBean(String json, JavaType clazz) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T toBean(byte[] json, Class<T> clazz) {
        if (json == null || json.length == 0) {
            return null;
        }
        if (String.class.isAssignableFrom(clazz)) {
            return (T) new String(json);
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T toBean(byte[] json, TypeReference<T> clazz) {
        if (json == null || json.length == 0) {
            return null;
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T toBean(byte[] json, JavaType clazz) {
        if (json == null || json.length == 0) {
            return null;
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> List<T> toBeanList(String json, Class<T> clazz) {
        val type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> List<T> toBeanList(String json, JavaType clazz) {
        val type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> List<T> toBeanList(byte[] json, Class<T> clazz) {
        val type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> List<T> toBeanList(byte[] json, JavaType clazz) {
        val type = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
        return toBean(json, type);
    }

    public static <T> T convertBean(Object obj, Class<T> clazz) {
        if (obj == null) {
            return null;
        }
        try {
            return objectMapper.convertValue(obj, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T convertBean(Object obj, TypeReference<T> clazz) {
        if (obj == null) {
            return null;
        }
        try {
            return objectMapper.convertValue(obj, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T convertBean(Object obj, JavaType clazz) {
        if (obj == null) {
            return null;
        }
        try {
            return objectMapper.convertValue(obj, clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static Map<String, Object> convertMap(Object obj) {
        return convertBean(obj, MAP_TYPE);
    }

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

}
