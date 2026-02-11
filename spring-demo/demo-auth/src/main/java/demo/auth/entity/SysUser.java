package demo.auth.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/07/15
 */
@Getter
@Setter
@TableName(value = "sys_user")
public class SysUser extends BaseSys {
    @Schema(description = "用户ID")
    @TableId("id")
    private Long id;
    public static final String ID = "id";
    @Schema(description = "用户昵称")
    @TableField("nick_name")
    private String nickName;
    public static final String NICK_NAME = "nick_name";
    @Schema(description = "用户邮箱")
    @TableField("email")
    private String email;
    public static final String EMAIL = "email";
    @Schema(description = "手机号码")
    @TableField("phone_number")
    private String phoneNumber;
    public static final String PHONE_NUMBER = "phone_number";
    @Schema(description = "头像路径")
    @TableField("avatar")
    private String avatar;
    public static final String AVATAR = "avatar";
    @Schema(description = "帐号状态（0正常 1停用）")
    @TableField("status")
    private Integer status;
    public static final String STATUS = "status";
    @Schema(description = "最后登录IP")
    @TableField("login_ip")
    private String loginIp;
    public static final String LOGIN_IP = "login_ip";
    @Schema(description = "最后登录时间")
    @TableField("login_date")
    private LocalDateTime loginDate;
    public static final String LOGIN_DATE = "login_date";

}
