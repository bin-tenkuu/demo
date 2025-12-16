package demo.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import demo.model.ResultModel;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;


/**
 * @author bin
 * @since 2022/12/16
 */
@Tag(name = "page")
@RestController
@RequestMapping("/page")
public class MybatisPageController {

    @Operation(summary = "默认")
    @GetMapping("/default")
    public ResultModel<List<String>> defaultPage(@Parameter(hidden = true) IPage<String> page) {
        return ResultModel.success(page);
    }

    @Operation(summary = "手动设置")
    @GetMapping("/seted")
    public ResultModel<List<String>> setedPage(@ParameterObject IPage<String> page) {
        return ResultModel.success(page);
    }

    @Operation(summary = "注解设置")
    @GetMapping("/annDefault")
    public ResultModel<List<String>> annDefaultPage(
            @ParameterObject IPage<String> page
    ) {
        return ResultModel.success(page);
    }
}
