package demo.controller;

import demo.entity.SysUserAuth;
import demo.model.LoginBody;
import demo.model.LoginUser;
import demo.model.ResultModel;
import demo.repository.SysUserAuthRepository;
import demo.repository.SysUserRepository;
import demo.service.TokenService;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.Random;

/**
 * @author bin
 * @since 2025/07/16
 */
@RestController()
@RequiredArgsConstructor
public class LoginController {
    private final SysUserRepository sysUserRepository;
    private final SysUserAuthRepository sysUserAuthRepository;
    private final PasswordEncoder passwordEncoder;
    private final TokenService tokenService;
    private final Random random = new Random();

    /**
     * 注册
     */
    @PostMapping("/register")
    public ResultModel<?> register(@RequestBody LoginBody body) {
        return ResultModel.success();
    }

    /**
     * 临时登陆申请录入，例如验证码等临时令牌登陆
     */
    @PostMapping("/tempLoginApply")
    public ResultModel<String> tempLoginApply(String username, String type, Long userId) {
        val userAuth = new SysUserAuth();
        userAuth.setUsername(username);
        userAuth.setType(type);
        userAuth.setUserId(userId);
        val password = String.valueOf(random.nextInt(111111, 999999));
        userAuth.setPassword(passwordEncoder.encode(password));
        SysUserAuthRepository.addExtryAuth(userAuth);
        return ResultModel.success(password);
    }

    @PostMapping("/login")
    public ResultModel<String> login(@RequestBody LoginBody body) {
        val userAuth = sysUserAuthRepository.findByUsername(body.getUsername());
        if (userAuth == null) {
            return ResultModel.fail("用户名或密码错误");
        }
        if (!passwordEncoder.matches(body.getPassword(), userAuth.getPassword())) {
            return ResultModel.fail("用户名或密码错误");
        }
        val sysUser = sysUserRepository.getById(userAuth.getUserId());
        if (sysUser == null) {
            return ResultModel.fail("用户不存在");
        }
        if (sysUser.getExpireTime() != null && sysUser.getExpireTime().isBefore(LocalDateTime.now())
            || sysUser.getStatus() == true
        ) {
            return ResultModel.fail("用户已过期或被禁用");
        }

        val loginUser = LoginUser.from(userAuth.getUsername(), userAuth.getPassword(), sysUser);
        val token = tokenService.createToken(loginUser);
        return ResultModel.success(token);
    }

    @PostMapping("/logout")
    public ResultModel<?> logout() {
        // 这里可以添加注销逻辑，例如清除用户会话等
        return ResultModel.success("登出成功");
    }
}
