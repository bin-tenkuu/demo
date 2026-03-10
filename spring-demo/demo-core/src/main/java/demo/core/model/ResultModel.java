package demo.core.model;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.Nullable;
import org.springframework.data.domain.Page;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// @author bin
/// @since 2023/01/31
@SuppressWarnings("unused")
@Getter
@Setter
@Schema(description = "返回")
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ResultModel<T> {
    private static final int SUCCESS_CODE = 200;
    private static final String SUCCESS_MESSAGE = "Success";
    private static final int FAIL_CODE = 500;
    private static final String FAIL_MESSAGE = "Fail";

    @Schema(description = "返回码，" + SUCCESS_CODE + "成功", example = SUCCESS_CODE + "")
    private int code;
    @Schema(description = "返回消息")
    private String message;
    @Nullable
    @Schema(description = "返回结果对象，该处是泛型")
    private T data;
    @Nullable
    @Schema(description = "分页对象")
    private PageModel page;
    @JsonIgnore
    @JsonAnyGetter
    private Map<String, Object> extra = new HashMap<>();

    public ResultModel() {
        this.code = SUCCESS_CODE;
        this.message = SUCCESS_MESSAGE;
    }

    public ResultModel(final int code, final String message) {
        this.code = code;
        this.message = message;
    }

    public ResultModel(@Nullable final T data) {
        this.code = SUCCESS_CODE;
        this.message = SUCCESS_MESSAGE;
        this.data = data;
    }

    public ResultModel(@Nullable final T data, @Nullable final PageModel page) {
        this.code = SUCCESS_CODE;
        this.message = SUCCESS_MESSAGE;
        this.data = data;
        this.page = page;
    }

    public ResultModel(int code, String message, @Nullable T data, @Nullable PageModel page) {
        this.code = code;
        this.message = message;
        this.data = data;
        this.page = page;
    }

    public void set(String key, Object value) {
        extra.put(key, value);
    }

    public Object put(String key, Object value) {
        return extra.put(key, value);
    }

    public static <T> ResultModel<T> success() {
        return new ResultModel<>();
    }

    public static <T> ResultModel<T> success(@Nullable final T data) {
        return new ResultModel<>(data);
    }

    public static <T> ResultModel<T> success(@Nullable final T data, @Nullable final PageModel page) {
        return new ResultModel<>(data, page);
    }

    public static <T> ResultModel<List<T>> success(IPage<T> page) {
        return success(page.getRecords(), page);
    }

    public static <T> ResultModel<List<T>> success(Page<T> page) {
        return success(page.getContent(), page);
    }

    public static <T> ResultModel<List<T>> success(List<T> list, IPage<?> page) {
        return ResultModel.success(list, new PageModel(
                (int) page.getCurrent(),
                (int) page.getSize(),
                (int) page.getPages(),
                page.getTotal()
        ));
    }

    public static <T> ResultModel<List<T>> success(List<T> list, Page<?> page) {
        return ResultModel.success(list, new PageModel(
                page.getNumber(),
                page.getSize(),
                page.getTotalPages(),
                page.getTotalElements()
        ));
    }

    public static <T> ResultModel<T> fail() {
        return new ResultModel<>(FAIL_CODE, FAIL_MESSAGE);
    }

    public static <T> ResultModel<T> fail(String message) {
        return new ResultModel<>(FAIL_CODE, message);
    }

    public static <T> ResultModel<T> fail(int code, String message) {
        return new ResultModel<>(code, message);
    }
}
