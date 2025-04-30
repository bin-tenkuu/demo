package demo.controller;

import demo.model.ResultModel;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author bin
 * @since 2025/04/30
 */
@Tag(name = "file")
@RestController
@RequestMapping("/file")
@RequiredArgsConstructor
public class FileController {
    @GetMapping("/download/{url}/**")
    public ResultModel<String> download(HttpServletRequest request, @PathVariable("url") String url) {
        val fullUrl = request.getRequestURI();
        val path = fullUrl.substring(fullUrl.indexOf(url) - 1);
        return ResultModel.success(path);
    }
}
