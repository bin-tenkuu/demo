package demo.config;

import demo.model.ResultModel;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolationException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.validation.ObjectError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.List;

@Slf4j
@RestControllerAdvice
public class ExceptionHandleAdvice  {

    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResultModel<String> HttpMessageNotReadableException(HttpServletRequest request,
            MissingServletRequestParameterException ex) {
        log.error(request.getRequestURI(), ex);
        String message = ex.getMessage();
        return ResultModel.fail(message);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResultModel<String> HttpMessageNotReadableException(HttpMessageNotReadableException ex) {
        String message = ex.getMessage();
        return ResultModel.fail(message);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResultModel<String> MethodArgumentNotValidException(MethodArgumentNotValidException ex) {
        BindingResult exceptions = ex.getBindingResult();
        // 判断异常中是否有错误信息，如果存在就使用异常中的消息，否则使用默认消息
        if (exceptions.hasErrors()) {
            List<ObjectError> errors = exceptions.getAllErrors();
            if (!errors.isEmpty()) {
                FieldError fieldError = (FieldError) errors.get(0);
                String message = fieldError.getDefaultMessage();
                return ResultModel.fail(message);
            }
        }
        String message = ex.getMessage();
        return ResultModel.fail(message);
    }

    @ExceptionHandler(NullPointerException.class)
    public ResultModel<String> NullPointerException(NullPointerException ex) {
        log.error("", ex);
        String message = ex.getMessage();
        return ResultModel.fail(message);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResultModel<String> IllegalArgumentException(IllegalArgumentException ex) {
        log.error("", ex);
        return ResultModel.fail(ex.getMessage());
    }

    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    public ResultModel<?> handleHttpMediaTypeNotSupportedException(HttpMediaTypeNotSupportedException ex) {
        log.error("", ex);
        return ResultModel.fail();
    }

    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResultModel<?> handleHttpRequestMethodNotSupportedException(HttpRequestMethodNotSupportedException ex) {
        log.error("", ex);
        return ResultModel.fail();
    }

    @ExceptionHandler(ConstraintViolationException.class)
    public ResultModel<?> handleConstraintViolationException(ConstraintViolationException ex) {
        log.error("", ex);
        return ResultModel.fail();
    }

}
