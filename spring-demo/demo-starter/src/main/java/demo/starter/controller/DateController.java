package demo.starter.controller;

import demo.core.model.RequestModel;
import demo.core.model.ResultModel;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;


/**
 * @author bin
 * @since 2022/12/28
 */
@Tag(name = "date")
@RestController
@RequestMapping("/v1/date")
public class DateController {
    @Operation(summary = "LocalDateTime")
    @GetMapping("/LocalDateTime")
    public ResultModel<LocalDateTime> getLocalDateTime(@RequestParam LocalDateTime time) {
        return ResultModel.success(time);
    }

    @Operation(summary = "LocalDate")
    @GetMapping("/LocalDate")
    public ResultModel<LocalDate> getLocalDateTime(@RequestParam LocalDate time) {
        return ResultModel.success(time);
    }

    @Operation(summary = "LocalTime")
    @GetMapping("/LocalTime")
    public ResultModel<LocalTime> getLocalDateTime(@RequestParam LocalTime time) {
        return ResultModel.success(time);
    }

    @Operation(summary = "String")
    @PostMapping("/String")
    public ResultModel<String> getString(@RequestBody RequestModel<LocalDateTime> model) {
        return ResultModel.success(model.getData().toString());
    }
}
