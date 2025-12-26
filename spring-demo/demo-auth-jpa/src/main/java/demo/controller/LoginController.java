package demo.controller;

import demo.entity.SysUserAuth;
import demo.model.ResultModel;
import demo.model.auth.LoginUser;
import demo.service.auth.TokenService;
import demo.util.SecurityUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/// @author bin
/// @since 2025/07/16
@Tag(name = "用户认证", description = "用户认证相关接口")
@Slf4j
@RestController
@RequiredArgsConstructor
public class LoginController {
    private final TokenService tokenService;

    @Operation(summary = "测试接口")
    @GetMapping
    public String hello() {
        return "Hello, Spring Boot!";
    }

    @Operation(summary = "用户注册")
    @PostMapping("/register")
    public ResultModel<?> register(@RequestBody SysUserAuth userAuth) {
        log.info("Register user: {}", userAuth);
        return ResultModel.success();
    }

    @Operation(summary = "临时登陆申请录入")
    @PostMapping("/tempLoginApply")
    public ResultModel<String> tempLoginApply(@RequestBody SysUserAuth userAuth) {
        var password = tokenService.addExtryAuth(userAuth);
        return ResultModel.success(password);
    }

    @Operation(summary = "用户登录")
    @PostMapping("/sys/login")
    public ResultModel<String> login(@RequestBody SysUserAuth body) {
        var token = tokenService.login(body);
        return ResultModel.success(token);
    }

    @Operation(summary = "用户登出")
    @PostMapping("/logout")
    public ResultModel<?> logout() {
        // 这里可以添加注销逻辑，例如清除用户会话等
        tokenService.delLoginUser(SecurityUtils.getLoginUser()
                .map(LoginUser::getToken)
                .orElse(null));
        return ResultModel.success("登出成功");
    }
}
