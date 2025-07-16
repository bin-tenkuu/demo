package demo.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import demo.entity.SysUser;
import demo.model.RequestModel;
import demo.model.ResultModel;
import demo.repository.SysUserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * @author bin
 * @since 2025/07/15
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/SysUser")
public class SysUserController {
    private final SysUserRepository repository;

    @PostMapping("/list")
    public ResultModel<List<SysUser>> list(IPage<SysUser> page) {
        IPage<SysUser> sysUserPage = repository.page(page);
        return ResultModel.success(sysUserPage);
    }

    @PostMapping("/save")
    public ResultModel<SysUser> save(@RequestBody SysUser sysUser) {
        boolean isSaved = repository.save(sysUser);
        if (isSaved) {
            return ResultModel.success(sysUser);
        } else {
            return ResultModel.fail("Failed to save user");
        }
    }

    @PostMapping("/update")
    public ResultModel<SysUser> update(@RequestBody SysUser sysUser) {
        boolean isUpdated = repository.updateById(sysUser);
        if (isUpdated) {
            return ResultModel.success(sysUser);
        } else {
            return ResultModel.fail("Failed to update user");
        }
    }

    @PostMapping("/delete")
    public ResultModel<Boolean> delete(@RequestBody RequestModel<Long> model) {
        boolean isDeleted = repository.removeById(model.getData());
        if (isDeleted) {
            return ResultModel.success(true);
        } else {
            return ResultModel.fail("Failed to delete user");
        }
    }
}
