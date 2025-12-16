package demo.model.auth;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import demo.entity.SysUser;
import demo.model.Query;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;

/// @author bin
/// @since 2023/05/30
@Getter
@Setter
@Schema(description = "系统用户查询")
public class SysUserQuery extends Query<SysUser> {

    /// 用户昵称
    private String nickName;

    /// 用户邮箱
    private String email;

    /// 手机号码
    private String phonenumber;

    /// 帐号状态（0正常 1停用）
    private String status;

    /// 最后登录IP
    private String loginIp;

    /// 最后登录时间
    private LocalDateTime loginDateStart;

    /// 最后登录时间
    private LocalDateTime loginDateEnd;

    /// 创建时间
    private LocalDateTime createTimeStart;

    private LocalDateTime createTimeEnd;

    /// 更新时间
    private LocalDateTime updateTimeStart;

    /// 更新时间
    private LocalDateTime updateTimeEnd;

    @Override
    public QueryWrapper<SysUser> toQuery() {
        return buildQuery()
                .like(StringUtils.hasLength(nickName), SysUser.NICK_NAME, nickName)
                .like(StringUtils.hasLength(email), SysUser.EMAIL, email)
                .like(StringUtils.hasLength(phonenumber), SysUser.PHONE_NUMBER, phonenumber)
                .eq(StringUtils.hasLength(status), SysUser.STATUS, status)
                .like(StringUtils.hasLength(loginIp), SysUser.LOGIN_IP, loginIp)
                .ge(loginDateStart != null, SysUser.LOGIN_DATE, loginDateStart)
                .le(loginDateEnd != null, SysUser.LOGIN_DATE, loginDateEnd)
                .ge(createTimeStart != null, SysUser.CREATE_TIME, createTimeStart)
                .le(createTimeEnd != null, SysUser.CREATE_TIME, createTimeEnd)
                .ge(updateTimeStart != null, SysUser.UPDATE_TIME, updateTimeStart)
                .le(updateTimeEnd != null, SysUser.UPDATE_TIME, updateTimeEnd)
                ;
    }
}
