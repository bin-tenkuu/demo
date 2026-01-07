create table sys_user
(
    id           bigint       not null
        primary key auto_increment,
    nick_name    varchar(255) not null,
    email        varchar(255),
    phone_number varchar(255),
    avatar       varchar(255),
    status       tinyint(1) default 0,
    login_ip     varchar(255),
    login_date   datetime,
    create_by    varchar(255),
    create_time  datetime   default current_timestamp,
    update_by    varchar(255),
    update_time  datetime   default current_timestamp
);
create table sys_user_auth
(
    username varchar(255) not null,
    type     varchar(255) not null,
    user_id  bigint       not null,
    password varchar(255) not null,
    primary key (username),
    unique (type, user_id)
);
create table sys_role
(
    id          bigint not null
        primary key auto_increment,
    role_key    varchar(255),
    role_name   varchar(255),
    role_sort   integer,
    status      tinyint(1),
    create_by   varchar(255),
    create_time datetime default current_timestamp,
    update_by   varchar(255),
    update_time datetime default current_timestamp
);
create table sys_menu
(
    id          bigint not null
        primary key auto_increment,
    menu_name   varchar(255),
    parent_id   bigint,
    order_num   integer,
    path        varchar(255),
    component   varchar(255),
    query       varchar(255),
    is_frame    tinyint(1),
    is_cache    tinyint(1),
    menu_type   varchar(255),
    visible     tinyint(1),
    status      tinyint(1),
    perms       varchar(255),
    icon        varchar(255),
    create_by   varchar(255),
    create_time datetime default current_timestamp,
    update_by   varchar(255),
    update_time datetime default current_timestamp,
    unique (menu_name)
);
create table sys_user_role
(
    user_id bigint not null,
    role_id bigint not null,
    primary key (user_id, role_id)
);
create table sys_role_menu
(
    role_id bigint not null,
    menu_id bigint not null,
    primary key (role_id, menu_id)
);
