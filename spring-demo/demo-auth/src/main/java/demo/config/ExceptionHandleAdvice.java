package demo.config;

import demo.model.ResultModel;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * @author bin
 * @since 2025/07/17
 */
@Slf4j
@RestControllerAdvice
public class ExceptionHandleAdvice {

    @ExceptionHandler(NullPointerException.class)
    public ResultModel<String> AuthenticationException(HttpServletRequest req, NullPointerException ex) {
        log.error("", ex);
        return ResultModel.fail("空指针异常，联系开发人员处理");
    }

}
